from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Iterable
from .analysis import tokenize

@dataclass
class Posting:
    doc_id: int
    positions: List[int]

class PositionalIndex:
    def __init__(self) -> None:
        self.postings: Dict[str, Dict[str, List[Posting]]] = defaultdict(lambda: defaultdict(list))
        self.vocab: Set[str] = set()
        self.fields: Set[str] = set()
        self.field_vocab: Dict[str, Set[str]] = defaultdict(set)
        self.docs: Dict[int, Dict[str, str]] = {}

    def add_document(self, doc_id: int, fields: Dict[str, str]) -> None:
        self.docs[doc_id] = fields
        for f, text in fields.items():
            self.fields.add(f)
            tokens = tokenize(text)
            positions_by_term: Dict[str, List[int]] = defaultdict(list)
            for i, tok in enumerate(tokens):
                positions_by_term[tok].append(i)
            for term, pos_list in positions_by_term.items():
                self.vocab.add(term)
                self.field_vocab[f].add(term)
                self.postings[term][f].append(Posting(doc_id, pos_list))

    def get_postings(self, term: str, field: Optional[str] = None) -> Dict[Tuple[int, str], List[int]]:
        out: Dict[Tuple[int, str], List[int]] = {}
        term_map = self.postings.get(term)
        if not term_map:
            return out

        if field:
            for p in term_map.get(field, []):
                out[(p.doc_id, field)] = p.positions
            return out

        for f, plist in term_map.items():
            for p in plist:
                out[(p.doc_id, f)] = p.positions

        return out

    def all_terms(self, field: Optional[str] = None) -> Iterable[str]:
        if field is None:
            return self.vocab
        return self.field_vocab.get(field, set())
