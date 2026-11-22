# config.py
# which metrics each benchmark reports, so results line up with that benchmark's published numbers

from dataclasses import dataclass
from typing import Dict, List, Tuple

from hybrid_rank.eval.gain_schemes import GAIN_SCHEMES


@dataclass(frozen=True)
class EvalConfig:
    gain: Dict[str, float]
    ndcg_ks: Tuple[int, ...] = (10, 50)
    p_k: int = 5
    recall_ks: Tuple[int, ...] = (50, 100)
    hit_ks: Tuple[int, ...] = ()
    # cnDCG / unlabelled@k need labelled non-relevant items to mean anything; binary answer-only
    # labels (STaRK) have none, so every non-answer would count as "unlabelled"
    condensed: bool = True
    tune_metrics: Tuple[str, str] = ("nDCG@10", "R@100")  # what tune_rrf.py prints grids for

    @property
    def depth(self) -> int:
        """How many results each query needs, i.e. the largest cut-off of any metric."""
        return max(*self.ndcg_ks, self.p_k, *self.recall_ks, *self.hit_ks)

    @property
    def metrics(self) -> List[str]:
        names = [f"nDCG@{k}" for k in self.ndcg_ks] + [f"P@{self.p_k}", "MRR"]
        names += [f"R@{k}" for k in self.recall_ks] + [f"Hit@{k}" for k in self.hit_ks]
        if self.condensed:
            names += [f"cnDCG@{self.ndcg_ks[0]}", f"unlabelled@{self.ndcg_ks[0]}"]
        return names

    @property
    def tested(self) -> List[str]:
        """Metrics worth a paired significance test (unlabelled@k is a diagnostic, not a quality score)."""
        return [m for m in self.metrics if not m.startswith("unlabelled")]


EVAL_CONFIGS = {
    "wands": EvalConfig(GAIN_SCHEMES["wands"]),
    # STaRK's paper reports Hit@1, Hit@5, Recall@20 and MRR; R@100 shows what a fusion / rerank pool holds
    "stark-amazon": EvalConfig(
        GAIN_SCHEMES["stark-amazon"],
        ndcg_ks=(10,),
        recall_ks=(20, 100),
        hit_ks=(1, 5),
        condensed=False,
        tune_metrics=("MRR", "R@20"),
    ),
}
