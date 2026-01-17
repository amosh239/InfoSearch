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
            docs = self._eval(node.left) & self._eval(node.right)
            res = set()
            
            if not isinstance(node.left, TermNode) or not isinstance(node.right, TermNode):
                return docs
                
            t1, t2 = node.left.term, node.right.term
            f1, f2 = node.left.field, node.right.field

            for doc_id in docs:
                p1_map = self.idx.get_postings(t1, f1)
                p2_map = self.idx.get_postings(t2, f2)
                
                fields = {f for d, f in p1_map if d == doc_id} & {f for d, f in p2_map if d == doc_id}
            
                for f in fields:
                    pos1 = p1_map[(doc_id, f)]
                    pos2 = p2_map[(doc_id, f)]
                    if self._check_pos(pos1, pos2, node.k, ordered=False):
                        res.add(doc_id)
                        break
            return res
        
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
        if not node.terms: return set()
        
        candidates = self._eval_term(TermNode(node.terms[0], field=node.field))
        for t in node.terms[1:]:
            candidates &= self._eval_term(TermNode(t, field=node.field))

        if not candidates: return set()

        results = set()
        for doc_id in candidates:
            base_posts = self.idx.get_postings(node.terms[0], node.field)
            common_fields = {f for (d, f) in base_posts if d == doc_id}
            
            for f in common_fields:
                valid_field = True
                curr_pos = base_posts[(doc_id, f)]
                
                for t in node.terms[1:]:
                    next_posts = self.idx.get_postings(t, f)
                    next_pos = next_posts.get((doc_id, f))
                    
                    if not next_pos or not self._check_pos(curr_pos, next_pos, 1, ordered=True):
                        valid_field = False
                        break
                    curr_pos = next_pos
                
                if valid_field:
                    results.add(doc_id)
                    break 
                    
        return results

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
    
    @staticmethod
    def _check_pos(p1: List[int], p2: List[int], k: int, ordered: bool) -> bool:
        i = j = 0
        while i < len(p1) and j < len(p2):
            diff = p2[j] - p1[i]
            if (ordered and diff == 1) or (not ordered and abs(diff) <= k):
                return True
            if p2[j] > p1[i]: i += 1
            else: j += 1
        return False