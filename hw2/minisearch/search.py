import json, os, re
from typing import List, Tuple
from markupsafe import Markup
from .index import PositionalIndex, tokenize
from .query import parse_query, TermNode, PhraseNode, NearNode, AndNode, OrNode, NotNode
from .ranking import L1Ranker, FeatureExtractor

def wildcard_to_regex(pattern: str) -> re.Pattern:
    esc = re.escape(pattern).replace(r'\*', '.*').replace(r'\?', '.')
    return re.compile('^' + esc + '$')

def edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2): return edit_distance(s2, s1)
    if len(s2) == 0: return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]

class Searcher:
    def __init__(self, index: PositionalIndex):
        self.idx = index
        w_path = os.path.join(os.path.dirname(__file__), 'weights.json')
        self.weights = json.load(open(w_path)) if os.path.exists(w_path) else [0.1, 1.0, 0.1]
        self.ranker = L1Ranker(FeatureExtractor(index), self.weights)

    def search(self, query: str) -> List[Tuple[int, float]]:
        try: ast = parse_query(query)
        except: ast = TermNode(query.lower())
        
        docs = self._eval(ast)
        if not docs: return []
        
        q_terms = [t for t in tokenize(query) if t.isalnum()]
        scored = [(d, self.ranker.score(d, q_terms)) for d in docs]
        return sorted(scored, key=lambda x: -x[1])

    def _eval(self, node) -> set[int]:
        if isinstance(node, TermNode):
            return self._eval_term(node)
        if isinstance(node, AndNode):
            return self._eval(node.left) & self._eval(node.right)
        if isinstance(node, OrNode):
            return self._eval(node.left) | self._eval(node.right)
        if isinstance(node, NotNode):
            return set(self.idx.docs) - self._eval(node.child)
        if isinstance(node, PhraseNode):
            return self._eval_phrase(node)
        if isinstance(node, NearNode):
             return self._eval(node.left) & self._eval(node.right)
        return set()

    def _eval_term(self, node: TermNode) -> set[int]:
        terms = {node.term}
        
        if node.wildcard:
            rx = wildcard_to_regex(node.term)
            terms = {t for t in self.idx.postings if rx.match(t)}
        elif node.fuzzy > 0:
            terms = {t for t in self.idx.postings if edit_distance(t, node.term) <= node.fuzzy}
        
        docs = set()
        for t in terms:
            for (did, _), _ in self.idx.get_postings(t, node.field).items():
                docs.add(did)
        return docs

    def _eval_phrase(self, node: PhraseNode) -> set[int]:
        res = None
        for t in node.terms:
            ds = self._eval_term(TermNode(t, field=node.field))
            res = ds if res is None else (res & ds)
        return res or set()

    def make_snippet(self, doc_id: int, query: str) -> str:
        text = str(self.idx.docs[doc_id].get('body', ''))
        q_words = [w for w in tokenize(query) if w.isalnum()]
        if not q_words: return text[:200]
        
        pattern = re.compile(f"({'|'.join(map(re.escape, q_words))})", re.I)
        match = pattern.search(text)
        start = max(0, match.start() - 50) if match else 0
        end = min(len(text), start + 200)
        snippet = text[start:end]
        return Markup(pattern.sub(r"<mark>\1</mark>", snippet))