from __future__ import annotations

import math
import heapq
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional, Set, Dict, List, Tuple, Callable


@dataclass
class QuorumConfig:
    activate_if_candidates_lt: int = 1000   # когда подключать quorum (внешний уровень)
    target: int = 1000                      # сколько кандидатов хотим получить
    cap: int = 5000                         # верхняя граница, чтобы не взрываться

    anchor_pool: int = 8                    # сколько редких термов брать в пул якорей
    max_df_frac: float = 0.20               # фильтр частотных термов (df <= frac * N)
    k_frac: float = 0.50                    # k = ceil(k_frac * n) для n>=4
    min_k: int = 2                          # минимальный k для длинных запросов
    max_universe: int = 20000               # ограничение на universe по union якорей


class QuorumCandidateGenerator:
    def __init__(self, index, tokenize_fn: Callable[[str], List[str]], config: Optional[QuorumConfig] = None):
        self.idx = index
        self.tokenize = tokenize_fn
        self.cfg = config or QuorumConfig()

        self.df = getattr(index, "df", {})
        self.N = len(getattr(index, "doc_ids", []))

    def generate(self, query: str) -> Set[int]:
        terms = self._query_terms(query)
        if len(terms) < 2:
            return set()

        info_terms = self._informative_terms(terms)
        k = self._k_value(len(info_terms))
        anchors = self._anchor_terms(info_terms)

        universe = self._universe_from_anchors(anchors)
        if not universe:
            return set()

        cnt, score = self._count_matches(info_terms, universe)
        out = {did for did, c in cnt.items() if c >= k}

        # добор до target по score (если нужно)
        if len(out) < self.cfg.target:
            need = min(self.cfg.target - len(out), len(score))
            if need > 0:
                for did, _ in heapq.nlargest(need, score.items(), key=lambda x: x[1]):
                    out.add(did)
                    if len(out) >= self.cfg.target:
                        break

        if len(out) > self.cfg.target:
            top = heapq.nlargest(
                self.cfg.target,
                ((did, score.get(did, 0.0)) for did in out),
                key=lambda x: x[1],
            )
            out = {did for did, _ in top}

        # cap
        if len(out) > self.cfg.cap:
            top = heapq.nlargest(
                self.cfg.cap,
                ((did, score.get(did, 0.0)) for did in out),
                key=lambda x: x[1],
            )
            out = {did for did, _ in top}

        return out


    def _query_terms(self, query: str) -> List[str]:
        terms = [t for t in self.tokenize(query) if t.isalnum()]
        return list(dict.fromkeys(terms))

    def _informative_terms(self, terms: List[str]) -> List[str]:
        if self.N <= 0:
            return terms
        max_df = int(self.cfg.max_df_frac * self.N)
        info = [t for t in terms if self.df.get(t, 0) <= max_df]
        return info if len(info) >= 2 else terms

    def _k_value(self, n: int) -> int:
        if n < 4:
            return max(1, n - 1)
        return max(self.cfg.min_k, math.ceil(self.cfg.k_frac * n))

    def _anchor_terms(self, terms: List[str]) -> List[str]:
        terms_sorted = sorted(terms, key=lambda t: self.df.get(t, 10**18))
        return terms_sorted[: min(self.cfg.anchor_pool, len(terms_sorted))]

    def _universe_from_anchors(self, anchors: List[str]) -> Set[int]:
        universe: Set[int] = set()
        for t in anchors:
            universe.update(self.idx.get_doc_ids(t, None))
            if len(universe) >= self.cfg.max_universe:
                break
        return universe

    def _idf(self, term: str) -> float:
        d = self.df.get(term, 0)
        if d <= 0 or self.N <= 0:
            return 0.0
        return math.log((self.N - d + 0.5) / (d + 0.5) + 1.0)

    def _count_matches(self, terms: List[str], universe: Set[int]) -> Tuple[Dict[int, int], Dict[int, float]]:
        cnt = defaultdict(int)
        score = defaultdict(float)

        for t in terms:
            w = self._idf(t)
            for did in self.idx.get_doc_ids(t, None):
                if did in universe:
                    cnt[did] += 1
                    score[did] += w

        return cnt, score
