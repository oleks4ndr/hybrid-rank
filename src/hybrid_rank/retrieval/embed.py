# embed.py
# dense retrieval: sentence-transformer encoder + exact cosine search over cached corpus embeddings

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from hybrid_rank.retrieval.base import top_k_ids
from hybrid_rank.schema import Listing

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
ENCODE_CHUNK = 20_000  # listings per saved chunk while encoding


@dataclass(frozen=True)
class EncoderSpec:
    name: str               # Hugging Face model id
    query_prefix: str = ""  # asymmetric models expect different prefixes on each side --
    doc_prefix: str = ""    # getting these wrong costs accuracy without raising any error


ENCODERS: Dict[str, EncoderSpec] = {
    "minilm": EncoderSpec("sentence-transformers/all-MiniLM-L6-v2"),
    "bge-small": EncoderSpec("BAAI/bge-small-en-v1.5", query_prefix=BGE_QUERY_PREFIX),
    "bge-base": EncoderSpec("BAAI/bge-base-en-v1.5", query_prefix=BGE_QUERY_PREFIX),
    "e5-base": EncoderSpec("intfloat/e5-base-v2", query_prefix="query: ", doc_prefix="passage: "),
}


def embeddings_dir(dataset: str) -> Path:
    return Path("data/processed") / dataset / "embeddings"


class DenseIndex:
    def __init__(self, listings: List[Listing], spec: EncoderSpec, cache_dir: Path, batch_size: int = 64):
        self.ids = [l.id for l in listings]
        self.spec = spec
        self.model = SentenceTransformer(spec.name)
        texts = [spec.doc_prefix + l.text for l in listings]
        self.cache_path = cache_dir / f"{spec.name.replace('/', '__')}_{self._cache_key(texts)}.npy"
        self.loaded_from_cache = self.cache_path.exists()
        self.doc_emb = self._load_or_encode(texts, batch_size)

    def _cache_key(self, texts: List[str]) -> str:
        # any change to model, truncation length, ids or text gets a new file instead of a stale one
        h = hashlib.sha1(f"{self.spec.name}|{self.model.max_seq_length}".encode())
        for lid, text in zip(self.ids, texts):
            h.update(f"\x00{lid}\x01{text}".encode())
        return h.hexdigest()[:12]

    def _load_or_encode(self, texts: List[str], batch_size: int) -> np.ndarray:
        if self.loaded_from_cache:
            emb = np.load(self.cache_path)
        else:
            emb = self._encode_resumable(texts, batch_size)
        assert emb.shape[0] == len(self.ids), f"{self.cache_path} has {emb.shape[0]} rows, expected {len(self.ids)}"
        return emb

    def _encode_resumable(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Encodes in chunks saved under <cache>.partial/, so an interrupted multi-hour encode
        (STaRK: ~1M long documents) resumes where it stopped instead of starting over."""
        partial_dir = self.cache_path.with_suffix(".partial")
        partial_dir.mkdir(parents=True, exist_ok=True)
        with tqdm(total=len(texts), desc=f"Encoding {self.spec.name}", unit="doc") as bar:
            for start in range(0, len(texts), ENCODE_CHUNK):
                chunk_path = partial_dir / f"{start:09d}.npy"
                chunk = texts[start : start + ENCODE_CHUNK]
                if not chunk_path.exists():
                    emb = self.model.encode(chunk, batch_size=batch_size, normalize_embeddings=True)
                    tmp = chunk_path.with_suffix(".tmp")
                    with open(tmp, "wb") as f:
                        np.save(f, np.asarray(emb, dtype=np.float32))
                    os.replace(tmp, chunk_path)  # a killed run never leaves a truncated chunk behind
                bar.update(len(chunk))
        emb = np.concatenate([np.load(p) for p in sorted(partial_dir.glob("*.npy"))])
        np.save(self.cache_path, emb)
        shutil.rmtree(partial_dir)
        return emb

    def search(self, query_text: str, top_k: int = 50, mask: Optional[np.ndarray] = None) -> List[str]:
        q = np.asarray(self.model.encode(self.spec.query_prefix + query_text, normalize_embeddings=True))
        scores = self.doc_emb @ q  # cosine similarity, since both sides are unit-normalized
        return top_k_ids(self.ids, scores, top_k, mask)
