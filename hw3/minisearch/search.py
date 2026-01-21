from collections import defaultdict
import re
from typing import List, Set, Tuple

from markupsafe import Markup

from .index import PositionalIndex, tokenize
from .query import (
    AndNode,
    NearNode,
    NotNode,
    OrNode,
    PhraseNode,
    TermNode,
    parse_query,
)
from .fast_ranking import FastRanker
from .quorum import QuorumCandidateGenerator, QuorumConfig

def wildcard_to_regex(pattern: str) -> re.Pattern:
    esc = re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".")
    return re.compile("^" + esc + "$")


def edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


class Searcher:
    def __init__(self, index: PositionalIndex, Ranker: FastRanker, QuorumGenerator: QuorumCandidateGenerator):
        self.idx = index
        self.ranker = Ranker
        self.quorum = QuorumGenerator

    def _set_quotum(self, new_quorumGenerator: QuorumCandidateGenerator):
        self.quorum = new_quorumGenerator

    def search(self, query: str) -> List[Tuple[int, float]]:
        try:
            ast = parse_query(query)
        except Exception:
            ast = TermNode(query.lower())

        docs = self._eval(ast)

        if self._is_simple_query(query) and len(docs) < self.quorum.cfg.activate_if_candidates_lt:
            docs |= self.quorum.generate(query)

        if not docs:
            return []

        q_terms = [t for t in tokenize(query) if t.isalnum()]
        q_terms = list(dict.fromkeys(q_terms))
        return self.ranker.get_scores(q_terms, candidate_ids=docs, top_k=10)


    def _is_simple_query(self, query: str) -> bool:
        specials = {"AND", "OR", "NOT", "NEAR", "(", ")", '"', ":", "*", "?", "~"}
        return not any(s in query for s in specials)
    

    def candidates(self, query: str) -> Set[int]:
        try:
            ast = parse_query(query)
        except Exception:
            ast = TermNode(query.lower())

        docs = self._eval(ast)
        if self._is_simple_query(query) and len(docs) < self.quorum.cfg.activate_if_candidates_lt:
            docs |= self.quorum.generate(query)
        return docs


    def _eval(self, node) -> Set[int]:
        if isinstance(node, TermNode):
            return self._eval_term(node)
        if isinstance(node, AndNode):
            return self._eval(node.left) & self._eval(node.right)
        if isinstance(node, OrNode):
            return self._eval(node.left) | self._eval(node.right)
        if isinstance(node, NotNode):
            return set(self.idx.doc_ids) - self._eval(node.child)
        if isinstance(node, PhraseNode):
            return self._eval_phrase(node)
        if isinstance(node, NearNode):
            return self._eval_near(node)
        return set()


    def _eval_term(self, node: TermNode) -> set[int]:
        # Normal term
        if not node.wildcard and node.fuzzy <= 0:
            return set(self.idx.get_doc_ids(node.term, node.field))

        # Wildcard
        if node.wildcard:
            rx = wildcard_to_regex(node.term)
            out = set()
            for t in self.idx.postings:
                if rx.match(t):
                    out.update(self.idx.get_doc_ids(t, node.field))
            return out

        # Fuzzy
        out = set()
        for t in self.idx.postings:
            if edit_distance(t, node.term) <= node.fuzzy:
                out.update(self.idx.get_doc_ids(t, node.field))
        return out


    def _eval_near(self, node: NearNode) -> Set[int]:
        docs = self._eval(node.left) & self._eval(node.right)
        if not docs:
            return set()
        if not isinstance(node.left, TermNode) or not isinstance(node.right, TermNode):
            return docs

        t1, t2 = node.left.term, node.right.term
        f1, f2 = node.left.field, node.right.field

        # If either side specifies a field: require NEAR within that field.
        field = f1 or f2
        fields = [field] if field else list(self.idx.fields)

        p1 = {f: self.idx.get_pos_map(t1, f) for f in fields}
        p2 = {f: self.idx.get_pos_map(t2, f) for f in fields}

        res: Set[int] = set()
        for doc_id in docs:
            for f in fields:
                pos1 = p1[f].get(doc_id)
                if not pos1:
                    continue
                pos2 = p2[f].get(doc_id)
                if not pos2:
                    continue
                if self._check_pos(pos1, pos2, node.k, ordered=False):
                    res.add(doc_id)
                    break
        return res


    def _eval_phrase(self, node: PhraseNode) -> Set[int]:
        if not node.terms:
            return set()

        field = node.field
        fields = [field] if field else list(self.idx.fields)

        pos = {t: {f: self.idx.get_pos_map(t, f) for f in fields} for t in node.terms}

        results: Set[int] = set()
        for f in fields:
            cand = set(pos[node.terms[0]][f].keys())
            for t in node.terms[1:]:
                cand &= set(pos[t][f].keys())
            if not cand:
                continue

            for doc_id in cand:
                curr_pos = pos[node.terms[0]][f][doc_id]
                ok = True
                for t in node.terms[1:]:
                    next_pos = pos[t][f][doc_id]
                    if not self._check_pos(curr_pos, next_pos, 1, ordered=True):
                        ok = False
                        break
                    curr_pos = next_pos
                if ok:
                    results.add(doc_id)

        return results


    @staticmethod
    def _check_pos(p1: List[int], p2: List[int], k: int, ordered: bool) -> bool:
        i = j = 0
        while i < len(p1) and j < len(p2):
            diff = p2[j] - p1[i]
            if ordered:
                if diff == k:
                    return True
                if diff > k:
                    i += 1
                else:
                    j += 1
            else:
                if abs(diff) <= k:
                    return True
                if p2[j] > p1[i]:
                    i += 1
                else:
                    j += 1
        return False


    def make_snippet(self, doc_id: int, query: str) -> str:
        text = str(self.idx.docs[doc_id].get("body", ""))
        q_words = [w for w in tokenize(query) if w.isalnum()]
        if not q_words:
            return text[:200]
        pattern = re.compile(f"({'|'.join(map(re.escape, q_words))})", re.I)
        match = pattern.search(text)
        start = max(0, match.start() - 50) if match else 0
        end = min(len(text), start + 200)
        snippet = text[start:end]
        return Markup(pattern.sub(r"<mark>\1</mark>", snippet))
