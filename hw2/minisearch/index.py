# index.py
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set, Iterable
import pickle # Для простой сериализации прямого индекса

from .analysis import tokenize
from .compression import compress_posting_list, decompress_posting_list

@dataclass
class Posting:
    doc_id: int
    positions: List[int]

class PositionalIndex:
    def __init__(self) -> None:
        # Обратный индекс: term -> field -> compressed_bytes
        self.postings: Dict[str, Dict[str, bytes]] = defaultdict(lambda: defaultdict(bytes))
        
        # Прямой индекс: doc_id -> field -> term_counts (для BM25 и фичей)
        self.direct_index: Dict[int, Dict[str, Counter]] = defaultdict(dict)
        
        # Статистика коллекции для BM25
        self.doc_lengths: Dict[int, int] = defaultdict(int) # doc_id -> total_len
        self.avg_doc_len: float = 0.0
        self.doc_count: int = 0
        
        self.vocab: Set[str] = set()
        self.fields: Set[str] = set()
        self.docs: Dict[int, Dict[str, str]] = {} # Сырые документы (для сниппетов)

    def add_document(self, doc_id: int, fields: Dict[str, str]) -> None:
        self.docs[doc_id] = fields
        doc_len = 0
        
        for f, text in fields.items():
            self.fields.add(f)
            tokens = tokenize(text)
            doc_len += len(tokens)
            
            # 1. Обновляем Прямой индекс (частоты слов)
            term_counts = Counter(tokens)
            self.direct_index[doc_id][f] = term_counts
            
            # Временная структура для позиций
            positions_by_term: Dict[str, List[int]] = defaultdict(list)
            for i, tok in enumerate(tokens):
                positions_by_term[tok].append(i)
                self.vocab.add(tok)
            
            # 2. Обновляем Обратный индекс (сразу сжимаем или накапливаем)
            # В реальной системе мы бы накапливали в памяти и сбрасывали на диск (flush).
            # Здесь для простоты будем хранить в памяти, но эмулировать сжатие.
            for term, pos_list in positions_by_term.items():
                # Примечание: В реальной жизни сжимают готовые списки целиком.
                # Т.к. мы добавляем по одному, "дописывание" в сжатый список сложное.
                # Для учебного примера мы упростим: будем хранить List[Posting] 
                # и сжимать только по запросу или "замораживать" индекс.
                # Но чтобы выполнить задание "сжатие", давай хранить "сырые" данные 
                # в списке self._temp_postings и иметь метод commit()
                pass 
                
        self.doc_lengths[doc_id] = doc_len
        self.doc_count += 1
        
        # ПЕРЕПИСЫВАЕМ логику добавления, чтобы не усложнять сжатие на лету:
        # Просто используем старый метод накопления, а сжатие добавим в отдельный метод.
        self._add_to_temp_index(doc_id, fields)

    def _add_to_temp_index(self, doc_id: int, fields: Dict[str, str]):
        if not hasattr(self, '_temp_postings'):
            self._temp_postings = defaultdict(lambda: defaultdict(list))
            
        for f, text in fields.items():
            tokens = tokenize(text)
            for i, tok in enumerate(tokens):
                # Формат временного постинга: (doc_id, position)
                self._temp_postings[tok][f].append((doc_id, i))

    def commit(self):
        """Превращает временные списки в сжатые."""
        self.avg_doc_len = sum(self.doc_lengths.values()) / self.doc_count if self.doc_count else 0
        
        for term, field_map in self._temp_postings.items():
            for field, raw_list in field_map.items():
                # raw_list = [(doc_id, pos), (doc_id, pos), ...]
                # Нам нужно сгруппировать по doc_id
                doc_map = defaultdict(list)
                for did, pos in raw_list:
                    doc_map[did].append(pos)
                
                # Сжимаем doc_ids
                doc_ids = sorted(doc_map.keys())
                compressed_docs = compress_posting_list(doc_ids)
                
                # Позиции можно тоже сжать (склеить все списки позиций), 
                # но для простоты оставим сжатие только doc_id
                # (чтобы декодер не усложнять слишком сильно для демо)
                
                # Сохраняем в self.postings. 
                # В боевой системе это был бы байтовый блоб: [doc_ids_blob][pos_blob]
                self.postings[term][field] = compressed_docs
                
                # Сохраняем позиции отдельно (в реальной жизни тоже сжали бы)
                if not hasattr(self, 'positions_store'): self.positions_store = {}
                self.positions_store[(term, field)] = doc_map

    def get_postings(self, term: str, field: Optional[str] = None) -> Dict[Tuple[int, str], List[int]]:
        """Распаковывает данные на лету."""
        out = {}
        # Проверяем временный индекс (если еще не commit)
        if hasattr(self, '_temp_postings') and term in self._temp_postings:
             # Логика работы с незакоммиченным индексом...
             # Для упрощения считаем, что всегда вызываем commit перед поиском
             pass

        term_map = self.postings.get(term)
        if not term_map: return out

        target_fields = [field] if field else term_map.keys()
        
        for f in target_fields:
            if f not in term_map: continue
            compressed_data = term_map[f]
            doc_ids = decompress_posting_list(compressed_data)
            
            # Достаем позиции
            # (в упрощенной схеме они лежат в positions_store)
            pos_map = self.positions_store.get((term, f), {})
            
            for did in doc_ids:
                if did in pos_map:
                    out[(did, f)] = pos_map[did]
        return out

    def all_terms(self, field: Optional[str] = None) -> Iterable[str]:
        if field is None: return self.vocab
        # Для простоты возвращаем все, фильтрация по полю тут сложнее из-за структуры
        return self.vocab