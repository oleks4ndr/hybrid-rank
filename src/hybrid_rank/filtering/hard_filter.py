# hard_filter.py
# Hard filters: the constraints a decomposed query states (brand: Nike, price <= 50) -> one boolean mask over
# the listings, which every retriever honours through search(..., mask=...).
#
# Catalog metadata is incomplete and inconsistently written, so the filter errs towards keeping listings:
#   - a listing with no value for a field is kept (missing="keep"; "drop" is the strict ablation)
#   - text matching ignores case, accents, punctuation, spacing and a plural "s"; a query value matches every
#     catalog value containing all of its words ("contains"), or for names, every value starting with them
#     ("prefix": "McFarlane" -> "McFarlane Toys")
#   - a value that matches nothing in the catalog is skipped, instead of leaving only listings without a value
#   - when fewer than min_candidates listings survive, constraints are dropped lowest priority first
#     (the adapter's field order)

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hybrid_rank.schema import FilterField

MISSING_POLICIES = ("keep", "drop")
MIN_CANDIDATES = 10
N_EXAMPLES = 12  # most common catalog values per text field, shown to the LLM
MAX_NAME_WORDS = 7
MIN_JOIN_LETTERS = 4

Key = Tuple[str, ...]


def _singular(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith("ss") else token


def normalize(value: Any) -> Key:
    """'The North-Face Jackets' -> ('north', 'face', 'jacket'). Query and catalog values go through the same
    function, so the crude singular only has to be consistent, not correct English."""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    tokens = [_singular(t) for t in re.findall(r"[a-z0-9]+", text)]
    if len(tokens) > 1 and tokens[0] == "the":
        tokens = tokens[1:]
    return tuple(tokens)


@dataclass
class FilterResult:
    mask: Optional[np.ndarray]                                # None: nothing applied, search every listing
    applied: Dict[str, Any] = field(default_factory=dict)     # field -> constraint that was applied
    unresolved: Dict[str, Any] = field(default_factory=dict)  # field -> values matching no catalog value
    relaxed: List[str] = field(default_factory=list)          # fields dropped to keep min_candidates
    n_candidates: int = 0                                     # listings the mask lets through


class FilterIndex:
    """Per-field lookup structures over the listings' filters, built once per corpus."""

    def __init__(self, filters: Sequence[Mapping[str, Any]], fields: Sequence[FilterField]):
        self.n = len(filters)
        self.fields = {f.name: f for f in fields}  # insertion order = priority
        self._rows: Dict[str, Dict[Key, np.ndarray]] = {}          # catalog value -> listing rows
        self._present: Dict[str, np.ndarray] = {}                  # listing has any value for the field
        self._by_token: Dict[str, Dict[str, List[Key]]] = {}       # word -> catalog values containing it
        self._by_joined: Dict[str, Dict[str, List[Key]]] = {}      # "trainingmask" -> ("training", "mask")
        self._examples: Dict[str, List[str]] = {}
        self._spelling: Dict[str, Dict[Key, str]] = {}             # catalog value -> first spelling seen
        self._numbers: Dict[str, np.ndarray] = {}                  # value per listing, nan if missing
        self._resolved: Dict[Tuple[str, Key], List[Key]] = {}
        for f in fields:
            if f.kind == "number":
                self._index_number(f.name, filters)
            else:
                self._index_text(f.name, filters)

    def _index_text(self, name: str, filters: Sequence[Mapping[str, Any]]) -> None:
        rows: Dict[Key, List[int]] = defaultdict(list)
        present = np.zeros(self.n, dtype=bool)
        parsed: Dict[str, Key] = {}  # each distinct raw string is normalized once
        counts: Counter = Counter()
        spelling: Dict[Key, str] = {}
        for i, listing_filters in enumerate(filters):
            raw = listing_filters.get(name)
            if raw is None:
                continue
            for value in raw if isinstance(raw, list) else [raw]:
                value = str(value)
                key = parsed.get(value)
                if key is None:
                    key = parsed[value] = normalize(value)
                if not key:
                    continue
                rows[key].append(i)
                present[i] = True
                counts[key] += 1
                spelling.setdefault(key, value)

        by_token: Dict[str, List[Key]] = defaultdict(list)
        by_joined: Dict[str, List[Key]] = defaultdict(list)
        for key in rows:
            for token in set(key):
                by_token[token].append(key)
            by_joined["".join(key)].append(key)
        self._rows[name] = {key: np.array(r, dtype=np.int64) for key, r in rows.items()}
        self._present[name] = present
        self._by_token[name], self._by_joined[name] = dict(by_token), dict(by_joined)
        self._spelling[name] = spelling
        self._examples[name] = [spelling[key] for key, _ in counts.most_common(N_EXAMPLES)]

    def _index_number(self, name: str, filters: Sequence[Mapping[str, Any]]) -> None:
        values = np.full(self.n, np.nan)
        for i, listing_filters in enumerate(filters):
            v = listing_filters.get(name)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                values[i] = v
        self._numbers[name] = values

    def catalog_notes(self) -> Dict[str, str]:
        """What each field's values look like in this catalog, for the decomposition prompt."""
        notes = {}
        for name, f in self.fields.items():
            if f.kind == "number":
                x = self._numbers[name][~np.isnan(self._numbers[name])]
                if len(x):
                    lo, mid, hi = np.percentile(x, [5, 50, 95])
                    notes[name] = f"Typical values {lo:g} to {hi:g}, median {mid:g}."
            elif self._examples[name]:
                notes[name] = "Most common values: " + ", ".join(self._examples[name]) + "."
        return notes

    def coverage(self, name: str) -> float:
        """Share of listings with a value for the field."""
        present = ~np.isnan(self._numbers[name]) if self.fields[name].kind == "number" else self._present[name]
        return float(present.mean()) if self.n else 0.0

    def mentions(self, name: str, text: str) -> List[Tuple[str, int]]:
        """Catalog values of a text field that the text names, as (spelling, number of listings), longest first.
        Names ("prefix" fields) must appear as a run of words. A name inside a longer one is dropped unless both
        start at the same word: "Creature skateboard" may mean Creature or Creature Skateboards. Other values need all their words anywhere in the text. High recall, not precision: ordinary words are
        often brand names too, so something has to judge which mentions are meant."""
        tokens = normalize(text)
        if self.fields[name].match == "prefix":
            spans: Dict[Tuple[int, int], List[Key]] = {}
            for i in range(len(tokens)):
                for j in range(i + 1, min(len(tokens), i + MAX_NAME_WORDS) + 1):
                    words = tokens[i:j]
                    # spacing variants ("training mask" / "TrainingMask") only between real words: joining
                    # short ones invents names ("has a" -> "Hasa")
                    joinable = len(words) == 1 or all(len(w) >= MIN_JOIN_LETTERS for w in words)
                    keys = [k for k in self._by_joined[name].get("".join(words), []) if k == words or joinable]
                    keys = [k for k in keys if len("".join(k)) > 1]
                    if keys:
                        spans[i, j] = keys
            outer = [s for s in spans if not any(o[0] < s[0] and s[1] <= o[1] for o in spans)]
            found = list(dict.fromkeys(k for s in outer for k in spans[s]))
        else:
            words = set(tokens)
            found = list(dict.fromkeys(
                k for w in dict.fromkeys(tokens) for k in self._by_token[name].get(w, []) if words.issuperset(k)
            ))
        found.sort(key=lambda k: -len(" ".join(k)))
        return [(self._spelling[name][k], len(self._rows[name][k])) for k in found]

    def resolve(self, name: str, value: Any) -> List[Key]:
        """The catalog values a query value matches."""
        query = normalize(value)
        if not query:
            return []
        cached = self._resolved.get((name, query))
        if cached is not None:
            return cached
        candidates = self._by_token[name].get(query[0], [])  # every match contains the first word
        if self.fields[name].match == "prefix":
            hits = [key for key in candidates if key[: len(query)] == query]
        else:
            words = set(query)
            hits = [key for key in candidates if words <= set(key)]
        seen = set(hits)
        hits += [key for key in self._by_joined[name].get("".join(query), []) if key not in seen]
        self._resolved[(name, query)] = hits
        return hits

    def _allowed(self, name: str, constraint: Any, missing: str) -> Tuple[Optional[np.ndarray], List[Any]]:
        """(listings one field's constraint allows, its values that match nothing); None if it can't apply."""
        if self.fields[name].kind == "number":
            lo, hi = constraint.get("min"), constraint.get("max")
            if lo is None and hi is None:
                return None, []
            x = self._numbers[name]
            with np.errstate(invalid="ignore"):  # nan compares False: missing values fail every bound
                ok = np.ones(self.n, dtype=bool)
                if lo is not None:
                    ok &= x >= lo
                if hi is not None:
                    ok &= x <= hi
            if missing == "keep":
                ok |= np.isnan(x)
            return ok, []

        keys: Dict[Key, None] = {}
        unresolved = []
        for value in constraint if isinstance(constraint, list) else [constraint]:
            hits = self.resolve(name, value)
            if hits:
                keys.update(dict.fromkeys(hits))
            else:
                unresolved.append(value)
        if not keys:
            return None, unresolved
        ok = np.zeros(self.n, dtype=bool)
        rows = self._rows[name]
        for key in keys:  # several values in one field are alternatives: "red or blue"
            ok[rows[key]] = True
        if missing == "keep":
            ok |= ~self._present[name]
        return ok, unresolved

    def apply(
        self, constraints: Mapping[str, Any], missing: str = "keep", min_candidates: int = MIN_CANDIDATES
    ) -> FilterResult:
        """AND of every field's constraint. Fields not in the index are ignored."""
        if missing not in MISSING_POLICIES:
            raise ValueError(f"missing must be one of {MISSING_POLICIES}, got {missing!r}")
        result = FilterResult(mask=None, n_candidates=self.n)
        active = []
        for name in self.fields:
            if name not in constraints:
                continue
            allowed, unresolved = self._allowed(name, constraints[name], missing)
            if unresolved:
                result.unresolved[name] = unresolved
            if allowed is not None:
                active.append((name, constraints[name], allowed))

        while active:
            mask = np.logical_and.reduce([allowed for _, _, allowed in active])
            n_candidates = int(mask.sum())
            if n_candidates >= min_candidates:
                result.mask, result.n_candidates = mask, n_candidates
                result.applied = {name: constraint for name, constraint, _ in active}
                break
            result.relaxed.append(active.pop()[0])  # too few left: give up the lowest-priority constraint
        return result
