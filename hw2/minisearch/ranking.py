import math
from collections import Counter

class FeatureExtractor:
    def __init__(self, index):
        self.idx = index
        self.k1 = 1.2
        self.b = 0.75

    def get_features(self, doc_id: int, query_terms: list[str]) -> list[float]:
        doc_len = self.idx.doc_lengths.get(doc_id, 0)
        direct = self.idx.direct_index.get(doc_id, {})
        avg_len = getattr(self.idx, 'avg_doc_len', 1) or 1
        
        bm25, found_terms = 0.0, set()
        
        for term in query_terms:
            tf = sum(counts.get(term, 0) for counts in direct.values())
            if tf > 0:
                found_terms.add(term)
                idf = 1.0 
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / avg_len)
                bm25 += idf * (tf * (self.k1 + 1)) / denom

        coverage = len(found_terms) / len(query_terms) if query_terms else 0
        proximity = coverage * 1.5 if coverage > 0.5 else 0.0

        return [bm25, coverage, proximity]

class L1Ranker:
    def __init__(self, extractor, weights):
        self.fe = extractor
        self.weights = weights

    def score(self, doc_id: int, query_terms: list[str]) -> float:
        feats = self.fe.get_features(doc_id, query_terms)
        return sum(w * f for w, f in zip(self.weights, feats))