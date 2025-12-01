from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Iterable
import html, re
import json
import os

from .index import PositionalIndex
from .query import (
    Node, TermNode, PhraseNode, NearNode, AndNode, OrNode, NotNode, parse_query
)
from .utils import wildcard_to_regex, edit_distance
from .analysis import tokenize

from .ranking import L1Ranker, FeatureExtractor

class Searcher:
    def __init__(self, index: PositionalIndex) -> None:
        self.idx = index
        self.fe = FeatureExtractor(index)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(base_dir, 'weights.json')

        try:
            with open(weights_path, 'r') as f:
                self.weights = json.load(f)
        except FileNotFoundError:
            self.weights = [0.1, 1.0, 0.1]
        
        self.ranker = L1Ranker(self.fe, self.weights)

    def _expand_term(self, t: TermNode) -> Set[str]:
        if not t.wildcard and t.fuzzy == 0:
            in_field = t.term in self.idx.all_terms(t.field) if t.field else t.term in self.idx.vocab
            return {t.term} if in_field else set()
        
        candidates = self.idx.all_terms(t.field)
        out: Set[str] = set()

        if t.wildcard:
            rx = wildcard_to_regex(t.term)
            for w in candidates:
                if rx.match(w): 
                    out.add(w)
        else:
            out = set(candidates)

        if t.fuzzy:
            kept: Set[str] = set()
            r = min(2, t.fuzzy)
            for w in out:
                if edit_distance(t.term, w, max_d=r) <= t.fuzzy:
                    kept.add(w)
            out = kept

        return set(sorted(out)[:50]) if len(out) > 50 else out

    def _eval_term(self, t: TermNode) -> Set[int]:
        docs = set()
        for term in self._expand_term(t):
            for (doc_id, _), _ in self.idx.get_postings(term, t.field).items():
                docs.add(doc_id)
        return docs

    def _eval_phrase_in_field(self, terms: List[str], field: str) -> Set[int]:
        if not terms: 
            return set()
        
        base_docs = None
        pos_maps = []
        for term in terms:
            pm = self.idx.get_postings(term, field)
            pos_maps.append(pm)
            ds = {doc for (doc, _f) in pm.keys()}
            base_docs = ds if base_docs is None else (base_docs & ds)
            if not base_docs: 
                return set()

        out = set()
        for d in base_docs:
            pos_candidates = set(pos_maps[0][(d, field)])
            for i in range(1, len(terms)):
                nxt = set(p - i for p in pos_maps[i][(d, field)])
                pos_candidates &= nxt
                if not pos_candidates: 
                    break
            if pos_candidates: 
                out.add(d)

        return out

    def _eval_phrase(self, pn: PhraseNode) -> Set[int]:
        if pn.field: 
            return self._eval_phrase_in_field(pn.terms, pn.field)
        
        res: Set[int] = set()
        for f in self.idx.fields:
            res |= self._eval_phrase_in_field(pn.terms, f)

        return res
    
    def _eval_near_pair_in_field(self, left_terms: set[str], right_terms: set[str], k: int, field: str) -> set[int]:
        out: set[int] = set()
        cand = self.docs_for_terms(left_terms, field) & self.docs_for_terms(right_terms, field)

        for d in cand:
            lp: list[int] = []
            rp: list[int] = []

            for t in left_terms:
                pm = self.idx.get_postings(t, field)  
                lp += pm.get((d, field), [])

            for t in right_terms:
                pm = self.idx.get_postings(t, field)
                rp += pm.get((d, field), [])

            if not lp or not rp:
                continue

            lp.sort(); rp.sort()

            i = j = 0
            while i < len(lp) and j < len(rp):
                diff = lp[i] - rp[j]
                if abs(diff) <= k:
                    out.add(d)
                    break
                if diff < 0: 
                    i += 1
                else: 
                    j += 1

        return out

    def _collect_words(self, node: Node) -> Set[str]:
        if isinstance(node, TermNode): 
            return self._expand_term(node)
        if isinstance(node, PhraseNode): 
            return set(node.terms)
        if isinstance(node, NotNode): 
            return set()
        if isinstance(node, (AndNode, OrNode, NearNode)):
            return self._collect_words(node.left) | self._collect_words(node.right)
        return set()

    def near_in_field_simple(self, left_terms: set[str], right_terms: set[str], k: int, field: str) -> set[int]:
        out = set()
        cand = self.docs_for_terms(left_terms, field) & self.docs_for_terms(right_terms, field)
        if not cand:
            return out

        for d in cand:
            lp: list[int] = []
            rp: list[int] = []
            for t in left_terms:
                pm = self.idx.get_postings(t, field)
                lp += pm.get((d, field), [])
            for t in right_terms:
                pm = self.idx.get_postings(t, field)
                rp += pm.get((d, field), [])

            if not lp or not rp:
                continue
            lp.sort()
            rp.sort()

            i = j = 0
            while i < len(lp) and j < len(rp):
                diff = lp[i] - rp[j]
                if abs(diff) <= k:
                    out.add(d); break
                if diff < 0: 
                    i += 1
                else:        
                    j += 1
                    
        return out

    def _field_of(self, node):
        from minisearch.query import TermNode, PhraseNode
        if isinstance(node, TermNode):  return node.field
        if isinstance(node, PhraseNode): return node.field
        return None

    def _eval_near(self, n):
        lwords = self._collect_words(n.left)
        rwords = self._collect_words(n.right)
        if not lwords or not rwords:
            return set()

        lf = self._field_of(n.left)
        rf = self._field_of(n.right)
        if lf and rf and lf != rf:
            return set()

        fields = [lf or rf] if (lf or rf) else self.idx.fields

        out: set[int] = set()
        for f in fields:
            out |= self.near_in_field_simple(lwords, rwords, n.k, f)
        return out


    def evaluate(self, node: Node) -> Set[int]:
        if isinstance(node, TermNode):   
            return self._eval_term(node)
        if isinstance(node, PhraseNode): 
            return self._eval_phrase(node)
        if isinstance(node, NearNode):   
            return self._eval_near(node)
        if isinstance(node, AndNode):   
            return self.evaluate(node.left) & self.evaluate(node.right)
        if isinstance(node, OrNode):     
            return self.evaluate(node.left) | self.evaluate(node.right)
        if isinstance(node, NotNode):    
            return set(self.idx.docs.keys()) - self.evaluate(node.child)
        return set()
    

    def docs_for_terms(self, terms, field):
        docs = set()
        for w in terms:
            for (doc_id, _f), _pos in self.idx.get_postings(w, field).items():
                docs.add(doc_id)
        return docs
    

    def search(self, query: str) -> List[Tuple[int, float]]:
        try: ast = parse_query(query)
        except Exception: ast = TermNode(query.lower())

        docs = self.evaluate(ast)
        if not docs: 
            return []
        
        query_terms = [t for t in tokenize(query) if t.isalnum()]
        scored_docs: List[Tuple[int, float]] = []
        for d in docs:
            score = self.ranker.score(d, query_terms)
            scored_docs.append((d, score))

        return sorted(scored_docs, key=lambda t: (-t[1], t[0]))


    def make_snippet(self, doc_id: int, query: str, max_len: int = 200) -> str:
        from markupsafe import Markup
        doc = self.idx.docs[doc_id]
        txt = '  \n'.join(f"{f}: {doc.get(f, '')}" for f in sorted(self.idx.fields)).strip()
        try:
            ast = parse_query(query)
            words = [w for w in self._collect_words(ast) if len(w) > 1]
        except Exception:
            words = []
        if not words:
            snippet = txt[:max_len] + ('…' if len(txt) > max_len else '')
            return Markup.escape(snippet)
        rx = re.compile("(" + "|".join(re.escape(w) for w in words) + ")", re.IGNORECASE)
        m = rx.search(txt)
        if not m:
            snippet = txt[:max_len] + ('…' if len(txt) > max_len else '')
            return Markup.escape(snippet)
        start = max(0, m.start() - max_len // 2)
        end = min(len(txt), start + max_len)
        snippet = txt[start:end]
        snippet = rx.sub(lambda mo: f"<mark>{Markup.escape(mo.group(0))}</mark>", snippet)
        if start > 0: snippet = '…' + snippet
        if end < len(txt): snippet = snippet + '…'
        return snippet
