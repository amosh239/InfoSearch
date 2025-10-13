from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Iterable
import html, re
from .index import PositionalIndex
from .query import (
    Node, TermNode, PhraseNode, NearNode, AndNode, OrNode, NotNode, parse_query
)
from .utils import wildcard_to_regex, edit_distance
from .analysis import tokenize

class Searcher:
    def __init__(self, index: PositionalIndex) -> None:
        self.idx = index

    # --- helpers ---
    def _expand_term(self, t: TermNode) -> Set[str]:
        if not t.wildcard and t.fuzzy == 0:
            in_field = t.term in self.idx.all_terms(t.field) if t.field else t.term in self.idx.vocab
            return {t.term} if in_field else set()
        candidates = self.idx.all_terms(t.field)
        out: Set[str] = set()
        if t.wildcard:
            rx = wildcard_to_regex(t.term)
            for w in candidates:
                if rx.match(w): out.add(w)
        else:
            out = set(candidates)
        if t.fuzzy:
            kept: Set[str] = set()
            r = min(2, t.fuzzy)
            for w in out:
                if edit_distance(t.term, w, max_d=r) <= t.fuzzy:
                    kept.add(w)
            out = kept
        return set(sorted(out)[:200]) if len(out) > 200 else out

    def _eval_term(self, t: TermNode) -> Set[int]:
        docs: Set[int] = set()
        for term in self._expand_term(t):
            for (doc_id, _f), _pos in self.idx.get_postings(term, t.field).items():
                docs.add(doc_id)
        return docs

    def _eval_phrase_in_field(self, terms: List[str], field: str) -> Set[int]:
        if not terms: return set()
        base_docs: Optional[Set[int]] = None
        pos_maps = []
        for term in terms:
            pm = self.idx.get_postings(term, field)
            pos_maps.append(pm)
            ds = {doc for (doc, _f) in pm.keys()}
            base_docs = ds if base_docs is None else (base_docs & ds)
            if not base_docs: return set()
        out: Set[int] = set()
        for d in base_docs:
            pos_candidates = set(pos_maps[0][(d, field)])
            for i in range(1, len(terms)):
                nxt = set(p - i for p in pos_maps[i][(d, field)])
                pos_candidates &= nxt
                if not pos_candidates: break
            if pos_candidates: out.add(d)
        return out

    def _eval_phrase(self, pn: PhraseNode) -> Set[int]:
        if pn.field: return self._eval_phrase_in_field(pn.terms, pn.field)
        res: Set[int] = set()
        for f in self.idx.fields:
            res |= self._eval_phrase_in_field(pn.terms, f)
        return res

    def _eval_near_pair_in_field(self, left_terms: Set[str], right_terms: Set[str], k: int, field: str) -> Set[int]:
        from collections import defaultdict as dd
        left_pos: Dict[int, List[int]] = dd(list)
        right_pos: Dict[int, List[int]] = dd(list)
        for t in left_terms:
            for (d, _f), pos in self.idx.get_postings(t, field).items():
                left_pos[d].extend(pos)
        for t in right_terms:
            for (d, _f), pos in self.idx.get_postings(t, field).items():
                right_pos[d].extend(pos)
        out: Set[int] = set()
        for d in (left_pos.keys() & right_pos.keys()):
            lp = sorted(left_pos[d]); rp = sorted(right_pos[d])
            i = j = 0
            while i < len(lp) and j < len(rp):
                if abs(lp[i] - rp[j]) <= k: out.add(d); break
                if lp[i] < rp[j]: i += 1
                else: j += 1
        return out

    def _collect_words(self, node: Node) -> Set[str]:
        if isinstance(node, TermNode): return self._expand_term(node)
        if isinstance(node, PhraseNode): return set(node.terms)
        if isinstance(node, NotNode): return set()
        if isinstance(node, (AndNode, OrNode, NearNode)):
            return self._collect_words(node.left) | self._collect_words(node.right)
        return set()

    # --- public ---
    def _eval_near(self, n: NearNode) -> Set[int]:
        def node_terms(n: Node):
            from typing import Tuple, Optional, Set
            if isinstance(n, TermNode) and not n.wildcard and n.fuzzy == 0:
                return (n.field, {n.term})
            if isinstance(n, PhraseNode) and n.terms:
                return (n.field, set(n.terms))
            return None
        lt = node_terms(n.left); rt = node_terms(n.right)
        if lt and rt:
            lf, lterms = lt; rf, rterms = rt
            if lf and rf and lf != rf: return set()
            fields: Iterable[str] = [lf or rf] if (lf or rf) else self.idx.fields
            res: Set[int] = set()
            for f in fields:
                res |= self._eval_near_pair_in_field(lterms, rterms, n.k, f)
            return res
        ldocs = self.evaluate(n.left); rdocs = self.evaluate(n.right)
        cand_docs = ldocs & rdocs
        out: Set[int] = set()
        for d in cand_docs:
            for f in self.idx.fields:
                toks = tokenize(self.idx.docs[d].get(f, ""))
                lw = self._collect_words(n.left); rw = self._collect_words(n.right)
                if not lw or not rw: continue
                lpos = [i for i,w in enumerate(toks) if w in lw]
                rpos = [i for i,w in enumerate(toks) if w in rw]
                i = j = 0; lpos.sort(); rpos.sort()
                while i < len(lpos) and j < len(rpos):
                    if abs(lpos[i] - rpos[j]) <= n.k: out.add(d); break
                    if lpos[i] < rpos[j]: i += 1
                    else: j += 1
        return out

    def evaluate(self, node: Node) -> Set[int]:
        if isinstance(node, TermNode):   return self._eval_term(node)
        if isinstance(node, PhraseNode): return self._eval_phrase(node)
        if isinstance(node, NearNode):   return self._eval_near(node)
        if isinstance(node, AndNode):    return self.evaluate(node.left) & self.evaluate(node.right)
        if isinstance(node, OrNode):     return self.evaluate(node.left) | self.evaluate(node.right)
        if isinstance(node, NotNode):    return set(self.idx.docs.keys()) - self.evaluate(node.child)
        return set()

    def search(self, query: str) -> List[Tuple[int, float]]:
        try: ast = parse_query(query)
        except Exception: ast = TermNode(query.lower())
        docs = self.evaluate(ast)
        if not docs: return []
        qw = self._collect_words(ast)
        scores: Dict[int, float] = defaultdict(float)
        for d in docs:
            for f in self.idx.fields:
                toks = tokenize(self.idx.docs[d].get(f, ""))
                scores[d] += sum(1 for w in toks if w in qw)
        return sorted(scores.items(), key=lambda t: (-t[1], t[0]))

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
