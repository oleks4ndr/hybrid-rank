# cross_encoder.py
# Local rerankers: a cross-encoder reads the query and one listing together and scores the pair.
# Scores are cached by the content of the pair, so a depth sweep or a second base run only scores new pairs.

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from hybrid_rank.llm import append_jsonl, read_jsonl

MAX_LENGTH = 512   # tokens for query + listing; longer listings lose their tail (on STaRK: reviews and Q&A)
CHUNK = 2_000      # pairs per saved chunk while scoring


@dataclass(frozen=True)
class RerankerSpec:
    name: str
    batch_size: int = 32


RERANKERS: Dict[str, RerankerSpec] = {
    "minilm": RerankerSpec("cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size=64),
    "bge-base": RerankerSpec("BAAI/bge-reranker-base"),
}


def rerank_cache_dir(dataset: str) -> Path:
    return Path("data/processed") / dataset / "rerank"


class CrossEncoderReranker:
    def __init__(self, spec: RerankerSpec, texts: Mapping[str, str], cache_dir: Path, model: Any = None):
        self.spec = spec
        self.texts = texts  # listing id -> text, for at least every listing to be reranked
        self.cache_path = cache_dir / f"{spec.name.replace('/', '__')}_len{MAX_LENGTH}.jsonl"
        self.scores: Dict[str, float] = {r["k"]: r["s"] for r in read_jsonl(self.cache_path) if "k" in r and "s" in r}
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder

            # raw logits: a sigmoid can saturate to 1.0 and turn the best candidates into ties
            self._model = CrossEncoder(self.spec.name, max_length=MAX_LENGTH, activation_fn=torch.nn.Identity())
        return self._model

    def _key(self, query_text: str, listing_id: str) -> str:
        return hashlib.sha1(f"{query_text}\x00{self.texts[listing_id]}".encode()).hexdigest()[:20]

    def missing(self, pairs: Iterable[Tuple[str, str]]) -> Dict[str, Tuple[str, str]]:
        """Uncached (query, listing id) pairs by cache key."""
        out: Dict[str, Tuple[str, str]] = {}
        for query_text, lid in pairs:
            key = self._key(query_text, lid)
            if key not in self.scores:
                out[key] = (query_text, self.texts[lid])
        return out

    def score(self, pairs: Iterable[Tuple[str, str]], progress: bool = False) -> int:
        """Score and cache every pair not already cached; returns how many were scored."""
        todo = list(self.missing(pairs).items())
        # similar lengths together wastes less padding
        todo.sort(key=lambda item: len(item[1][1]))
        for start in range(0, len(todo), CHUNK):
            chunk = todo[start : start + CHUNK]
            values = self.model.predict([list(pair) for _, pair in chunk], batch_size=self.spec.batch_size,
                                        show_progress_bar=False)
            records = [{"k": key, "s": float(v)} for (key, _), v in zip(chunk, values)]
            append_jsonl(self.cache_path, records)
            self.scores.update((r["k"], r["s"]) for r in records)
            if progress:
                print(f"  scored {start + len(chunk):,} / {len(todo):,} pairs", flush=True)
        return len(todo)

    def rerank(self, query_text: str, listing_ids: List[str]) -> List[str]:
        self.score((query_text, lid) for lid in listing_ids)
        # sorted() is stable: equal scores keep the retriever's order
        return sorted(listing_ids, key=lambda lid: -self.scores[self._key(query_text, lid)])
