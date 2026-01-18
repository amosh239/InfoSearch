import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict

class FastRanker:
    def __init__(self, index, weights=None):
        # weights: [overlap_weight, bm25_weight]
        # Обычно BM25 самодостаточен, поэтому overlap можно уменьшить или убрать
        self.weights = weights if weights else [1.0, 1.0]
        
        # Параметры BM25
        self.k1 = 1.2
        self.b = 0.75
        
        print("Building BM25 matrix...")
        
        # 1. Документы
        if hasattr(index, 'docs'):
            self.doc_ids = list(index.docs.keys())
        elif hasattr(index, 'documents'):
             self.doc_ids = list(index.documents.keys())
        else:
            raise AttributeError("Index has no 'docs' attribute")

        self.doc_to_idx = {did: i for i, did in enumerate(self.doc_ids)}
        self.n_docs = len(self.doc_ids)
        
        # 2. Длины документов (Нужны для BM25)
        # index.doc_lengths - словарь {doc_id: length}
        if not hasattr(index, 'doc_lengths'):
            raise AttributeError("Index needs 'doc_lengths' for BM25")
            
        self.avg_dl = sum(index.doc_lengths.values()) / self.n_docs if self.n_docs > 0 else 1.0
        
        # Превращаем длины в массив, где индекс соответствует строке матрицы
        self.doc_lens_arr = np.zeros(self.n_docs)
        for did, length in index.doc_lengths.items():
            if did in self.doc_to_idx:
                self.doc_lens_arr[self.doc_to_idx[did]] = length

        # 3. Термины
        self.terms = list(index.postings.keys())
        self.term_to_idx = {t: i for i, t in enumerate(self.terms)}
        self.n_terms = len(self.terms)
        
        # 4. Сбор статистики из кэша
        if not hasattr(index, '_pos_cache') or not index._pos_cache:
            print("Warning: _pos_cache missing")
            cache_source = {}
        else:
            cache_source = index._pos_cache

        # Агрегация TF (term -> doc -> count)
        term_doc_counts = defaultdict(lambda: defaultdict(int))
        for (term, field), doc_map in cache_source.items():
            if term not in self.term_to_idx: continue
            t_idx = self.term_to_idx[term]
            for doc_id, positions in doc_map.items():
                if doc_id not in self.doc_to_idx: continue
                d_idx = self.doc_to_idx[doc_id]
                term_doc_counts[t_idx][d_idx] += len(positions)

        # 5. Расчет BM25 и заполнение матрицы
        rows = []
        cols = []
        data_bm25 = []
        
        for t_idx, doc_counts in term_doc_counts.items():
            df = len(doc_counts)
            if df == 0: continue
            
            # IDF (BM25 версия)
            idf = np.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            
            for d_idx, tf in doc_counts.items():
                doc_len = self.doc_lens_arr[d_idx]
                
                # Формула BM25
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_dl))
                bm25_val = idf * (numerator / denominator)
                
                rows.append(d_idx)
                cols.append(t_idx)
                data_bm25.append(bm25_val)

        # 6. Создаем матрицы
        self.matrix_bm25 = csr_matrix((data_bm25, (rows, cols)), shape=(self.n_docs, self.n_terms))
        
        # Бинарная для Overlap (можно отключить вес, если не нужно)
        data_bin = np.ones(len(rows))
        self.matrix_bin = csr_matrix((data_bin, (rows, cols)), shape=(self.n_docs, self.n_terms))
        
        print(f"BM25 Matrix built: {self.n_docs} docs x {self.n_terms} terms")

    def get_scores(self, query_terms, candidate_ids=None):
        term_idxs = [self.term_to_idx[t] for t in query_terms if t in self.term_to_idx]
        if not term_idxs: return {}

        # Считаем
        overlap_scores = np.array(self.matrix_bin[:, term_idxs].sum(axis=1)).flatten()
        bm25_scores = np.array(self.matrix_bm25[:, term_idxs].sum(axis=1)).flatten()
        
        total_scores = (self.weights[0] * overlap_scores) + (self.weights[1] * bm25_scores)
        
        results = {}
        if candidate_ids:
            # Оптимизация: если кандидатов МАЛО, идем циклом
            # Если кандидатов МНОГО (> 1000), быстрее векторизованно
            # Но пока оставим цикл для надежности
            for doc_id in candidate_ids:
                if doc_id in self.doc_to_idx:
                    idx = self.doc_to_idx[doc_id]
                    if total_scores[idx] > 0:
                        results[doc_id] = total_scores[idx]
        else:
            non_zero = total_scores.nonzero()[0]
            for idx in non_zero:
                doc_id = self.doc_ids[idx]
                results[doc_id] = total_scores[idx]
                
        return results