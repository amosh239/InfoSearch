# ranking.py
import math
from collections import Counter
from typing import List, Dict, Set
from .index import PositionalIndex
from .analysis import tokenize

class FeatureExtractor:
    def __init__(self, index: PositionalIndex):
        self.idx = index
        self.k1 = 1.2
        self.b = 0.75

    def get_features(self, doc_id: int, query_terms: List[str]) -> List[float]:
        """
        Возвращает вектор фичей для пары (doc_id, query).
        Фичи:
        0. BM25 score (сумма по полям)
        1. Полное вхождение фразы (Exact Phrase Match count)
        2. Покрытие терминов (сколько слов из запроса есть в документе)
        3. Минимальное расстояние между словами (Proximity) - упрощенно
        """
        doc_len = self.idx.doc_lengths[doc_id]
        avg_len = self.idx.avg_doc_len
        doc_struct = self.idx.direct_index[doc_id] # Прямой индекс: field -> Counter
        
        bm25_score = 0.0
        exact_matches = 0
        terms_found = set()
        
        # --- 1. BM25 & TF-IDF ---
        for term in query_terms:
            # Считаем TF по всем полям (простая сумма)
            tf = 0
            for field, counts in doc_struct.items():
                tf += counts.get(term, 0)
            
            if tf > 0:
                terms_found.add(term)
                
                # IDF (упрощенно: считаем DF по 'title' и 'body' вместе)
                # В реальном коде DF лучше прекалькулировать в индексе
                df = 0
                for f in self.idx.fields:
                    if term in self.idx.postings and f in self.idx.postings[term]:
                         # Внимание: тут мы делаем decompress каждый раз, это медленно для обучения!
                         # В продакшене DF хранят отдельно в словаре.
                         # Для демо допустимо.
                         # Лучше добавь self.df = defaultdict(int) в Index и считай при индексации.
                         pass
                # Эмуляция DF (для примера берем 1, если лень считать, но лучше считать честно)
                # Давай предположим, что df передадим или посчитаем "на лету" грубо
                df = 1 # Заглушка, надо брать из индекса
                
                N = self.idx.doc_count
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                
                score = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / avg_len))
                bm25_score += score

        # --- 2. Proximity (Минимальное окно) ---
        # Нужно найти позиции всех слов запроса и найти минимальный span
        # Это сложно без распаковки позиций. Если используем search.py, там это есть.
        # Пока поставим заглушку или простую эвристику.
        proximity_score = 1.0 / (len(query_terms) + 1) # Заглушка

        return [
            bm25_score,
            len(terms_found) / len(query_terms) if query_terms else 0, # Query Coverage
            proximity_score
        ]

class L1Ranker:
    def __init__(self, feature_extractor: FeatureExtractor, weights: List[float]):
        self.fe = feature_extractor
        self.weights = weights # Веса для [bm25, coverage, proximity]

    def score(self, doc_id: int, query_terms: List[str]) -> float:
        feats = self.fe.get_features(doc_id, query_terms)
        return sum(w * f for w, f in zip(self.weights, feats))