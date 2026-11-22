# bm25.py
# BM25 (k1=1.5, b=0.75) with every (term, listing) weight precomputed into a sparse matrix, so a query only
# touches the listings that contain its terms. rank_bm25 keeps a dict per listing and loops over all of
# them in Python for each query term, which doesn't scale to STaRK's ~1M long documents.
#
# Two idf variants:
#   okapi  -- rank_bm25's BM25Okapi, bit for bit: log((N-df+.5)/(df+.5)), and terms in over half the
#             listings (negative idf) floored to 0.25 x mean idf. With a large vocabulary that floor is
#             high: on STaRK "the", "for", "and" get idf 3.17, more than informative words like "looking".
#   lucene -- log(1 + (N-df+.5)/(df+.5)), never negative, so no floor. Used by Lucene / Pyserini and by
#             bm25s, which STaRK's own BM25 baseline runs on.

import hashlib
import math
import os
import re
from array import array
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy import sparse
from tqdm import tqdm

from hybrid_rank.retrieval.base import top_k_ids
from hybrid_rank.schema import Listing

K1, B, EPSILON = 1.5, 0.75, 0.25
IDF_VARIANTS = ("lucene", "okapi")
VERSION = "k1=1.5-b=0.75-eps=0.25-tok=[a-z0-9]+"  # part of the cache key: change it if scoring changes


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_dir(dataset: str) -> Path:
    return Path("data/processed") / dataset / "bm25"


class BM25Index:
    def __init__(self, listings: List[Listing], cache_dir: Optional[Path] = None, idf: str = "lucene"):
        if idf not in IDF_VARIANTS:
            raise ValueError(f"idf must be one of {IDF_VARIANTS}, got {idf!r}")
        self.ids = [l.id for l in listings]
        self.idf_variant = idf
        self.cache_path = cache_dir / f"bm25-{idf}_{self._cache_key(listings)}.npz" if cache_dir else None
        self.loaded_from_cache = self.cache_path is not None and self.cache_path.exists()
        if self.loaded_from_cache:
            self._load()
        else:
            self._build(listings)
            if self.cache_path is not None:
                self._save()

    def _cache_key(self, listings: List[Listing]) -> str:
        h = hashlib.sha1(f"{self.idf_variant}|{VERSION}".encode())
        for l in listings:
            h.update(f"\x00{l.id}\x01{l.text}".encode())
        return h.hexdigest()[:12]

    def _build(self, listings: List[Listing]) -> None:
        n = len(listings)
        vocab: dict = {}  # insertion order = first occurrence in the corpus, as in rank_bm25
        term_ids, tfs = array("i"), array("i")  # one entry per distinct (listing, term); compact, unlike lists
        n_terms = np.zeros(n, dtype=np.int64)
        doc_len = np.zeros(n, dtype=np.int64)
        for i, listing in enumerate(tqdm(listings, desc="Indexing BM25", disable=n < 100_000)):
            counts = Counter(_tokenize(listing.text))
            n_terms[i], doc_len[i] = len(counts), sum(counts.values())
            term_ids.extend(vocab.setdefault(t, len(vocab)) for t in counts)
            tfs.extend(counts.values())
        t = np.frombuffer(term_ids, dtype=np.int32)
        tf = np.frombuffer(tfs, dtype=np.int32)
        doc = np.repeat(np.arange(n, dtype=np.int32), n_terms)

        # Everything below follows rank_bm25's order of operations so okapi scores match it bit for bit:
        # math.log, a plain running sum in vocab order (Python 3.12+ sum() compensates, which differs),
        # and the weight expression evaluated in the same order (only commutative steps are reordered).
        df = np.bincount(t, minlength=len(vocab))
        if self.idf_variant == "okapi":
            idf = np.array([math.log(n - d + 0.5) - math.log(d + 0.5) for d in df.tolist()])
            idf_sum = 0.0
            for value in idf.tolist():
                idf_sum += value
            if len(idf):
                idf[idf < 0] = EPSILON * (idf_sum / len(idf))
        else:
            idf = np.log1p((n - df + 0.5) / (df + 0.5))
        avgdl = doc_len.sum() / n

        # idf * (tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))), one in-place step at a time so a
        # ~1M-listing corpus never holds more than a few nnz-sized arrays at once
        norm = doc_len.astype(np.float64)[doc]
        norm *= B
        norm /= avgdl
        norm += 1 - B
        norm *= K1
        norm += tf
        weights = tf * (K1 + 1)
        weights /= norm
        del norm
        weights *= idf[t]
        # rows = terms, so one term's postings are a contiguous slice
        matrix = sparse.csr_matrix((weights, (t, doc)), shape=(len(vocab), n))
        del weights, t, tf, doc
        self.data, self.indices, self.indptr = matrix.data, matrix.indices, matrix.indptr
        self.vocab = vocab

    def _save(self) -> None:
        assert self.cache_path is not None
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp.npz")
        np.savez(tmp, data=self.data, indices=self.indices, indptr=self.indptr)
        # terms are [a-z0-9]+, so newline-separated text is unambiguous
        self.cache_path.with_suffix(".vocab.txt").write_text("\n".join(self.vocab))
        os.replace(tmp, self.cache_path)  # the .npz appears last, so a half-written cache is never "found"

    def _load(self) -> None:
        assert self.cache_path is not None
        with np.load(self.cache_path) as z:
            self.data, self.indices, self.indptr = z["data"], z["indices"], z["indptr"]
        terms = self.cache_path.with_suffix(".vocab.txt").read_text().split("\n")
        self.vocab = {term: i for i, term in enumerate(terms)}

    def get_scores(self, query_text: str) -> np.ndarray:
        scores = np.zeros(len(self.ids))
        for term in _tokenize(query_text):  # a repeated query term counts again, as in rank_bm25
            j = self.vocab.get(term)
            if j is not None:
                lo, hi = self.indptr[j], self.indptr[j + 1]
                scores[self.indices[lo:hi]] += self.data[lo:hi]  # a term's listings are unique, so += is safe
        return scores

    def search(self, query_text: str, top_k: int = 50, mask: Optional[np.ndarray] = None) -> List[str]:
        scores = self.get_scores(query_text)
        # a listing sharing no term with the query scores 0 -- it isn't a lexical match, so don't return it
        matched = scores > 0
        return top_k_ids(self.ids, scores, top_k, matched if mask is None else matched & mask)
