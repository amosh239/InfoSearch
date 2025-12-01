from collections import defaultdict, Counter
from typing import Dict, List, Optional
from .analysis import tokenize
from .compression import compress_posting_list, decompress_posting_list

class PositionalIndex:
    def __init__(self):
        self.postings = defaultdict(dict) 
        self.direct_index = {}           
        self.doc_lengths = {}
        self.docs = {}
        
        self._raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    def add_document(self, doc_id: int, fields: Dict[str, str]):
        self.docs[doc_id] = fields
        self.direct_index[doc_id] = {}
        total_len = 0
        
        for field, text in fields.items():
            tokens = tokenize(text)
            total_len += len(tokens)
            self.direct_index[doc_id][field] = Counter(tokens)
            
            for pos, term in enumerate(tokens):
                self._raw[term][field][doc_id].append(pos)
        
        self.doc_lengths[doc_id] = total_len

    def commit(self):
        self.avg_doc_len = sum(self.doc_lengths.values()) / len(self.doc_lengths) if self.doc_lengths else 0
        
        for term, field_map in self._raw.items():
            for field, doc_map in field_map.items():
                ids = sorted(doc_map.keys())
                self.postings[term][field] = compress_posting_list(ids)
                
                if not hasattr(self, '_pos_cache'): self._pos_cache = {}
                self._pos_cache[(term, field)] = doc_map
        
        self._raw.clear()

    def get_postings(self, term: str, field: str = None):
        res = {}
        target_fields = [field] if field else self.postings.get(term, {}).keys()
        
        for f in target_fields:
            if compressed := self.postings[term].get(f):
                doc_ids = decompress_posting_list(compressed)
                pos_map = getattr(self, '_pos_cache', {}).get((term, f), {})
                for did in doc_ids:
                    res[(did, f)] = pos_map.get(did, [])
        return res

    @property
    def fields(self):
        return {f for d in self.docs.values() for f in d}