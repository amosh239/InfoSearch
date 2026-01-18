import re
from collections import Counter, defaultdict
from typing import Dict, List, Set

_TOKEN_RE = re.compile(r"[\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    return text.lower()


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.split(normalize(text)) if t]


class PositionalIndex:
    def __init__(self):
        self.postings = defaultdict(dict)

        self._pos_cache = {}
        self.docs: Dict[int, Dict[str, str]] = {}
        self.doc_ids: Set[int] = set()
        self._fields_set: Set[str] = set()

        self.direct_index: Dict[int, Dict[str, Counter]] = {}
        self.doc_lengths: Dict[int, int] = {}
        self.avg_doc_len: float = 0.0

        self._raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # self.postings_all = {}
        # self.df = {}


    def add_document(self, doc_id: int, fields: Dict[str, str]):
        self.docs[doc_id] = fields
        self.doc_ids.add(doc_id)
        self._fields_set.update(fields.keys())

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
        self.avg_doc_len = (
            sum(self.doc_lengths.values()) / len(self.doc_lengths)
            if self.doc_lengths
            else 0.0
        )

        for term, field_map in self._raw.items():
            for field, doc_map in field_map.items():
                ids = sorted(doc_map.keys())
                self.postings[term][field] = ids
                self._pos_cache[(term, field)] = doc_map

        self._raw.clear()

    def get_doc_ids(self, term: str, field: str | None = None) -> List[int]:
        if field is not None:
            return list(self.postings.get(term, {}).get(field, []))

        out: List[int] = []
        for ids in self.postings.get(term, {}).values():
            out.extend(ids)
        return out

    def get_pos_map(self, term: str, field: str) -> Dict[int, List[int]]:
        return self._pos_cache.get((term, field), {})

    @property
    def fields(self) -> Set[str]:
        return set(self._fields_set)
