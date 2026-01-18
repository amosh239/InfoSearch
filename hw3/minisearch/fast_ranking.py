import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict


class FastRanker:
    def __init__(self, index, weights=[1.0, 1.0]):
        self.index = index
        self.weights = weights
        self.k1 = 1.2
        self.b = 0.75

        self._init_docs()
        self._init_terms()
        self._build_tf_from_positions()
        self._build_matrices()

        del self.term_doc_tf


    def _init_docs(self):
        self.doc_ids = sorted(self.index.doc_ids)
        self.n_docs = len(self.doc_ids)

        self.doc_to_idx = {did: i for i, did in enumerate(self.doc_ids)}

        self.doc_lens_arr = np.array(
            [float(self.index.doc_lengths.get(did, 0)) for did in self.doc_ids],
            dtype=np.float32,
        )

        self.avg_dl = float(self.doc_lens_arr.mean()) if self.n_docs else 1.0
        if self.avg_dl <= 0:
            self.avg_dl = 1.0

    def _init_terms(self):
        self.terms = list(self.index.postings.keys())
        self.n_terms = len(self.terms)

        self.term_to_idx = {t: i for i, t in enumerate(self.terms)}
        self.idf = np.zeros(self.n_terms, dtype=np.float32)


    def _build_tf_from_positions(self):
        pos_cache = self.index._pos_cache
        assert pos_cache, "BM25 matrix requires positions cache (_pos_cache)."

        term_doc_tf = defaultdict(lambda: defaultdict(int))

        for (term, _field), doc_map in pos_cache.items():
            t_idx = self.term_to_idx.get(term)
            if t_idx is None:
                continue
            per_doc = term_doc_tf[t_idx]
            for doc_id, positions in doc_map.items():
                d_idx = self.doc_to_idx.get(doc_id)
                if d_idx is not None:
                    per_doc[d_idx] += len(positions)

        self.term_doc_tf = term_doc_tf

    def _bm25(self, tf: int, dl: float, idf: float) -> float:
        denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_dl))
        return float(idf * (tf * (self.k1 + 1.0) / denom))

    def _build_matrices(self):
        rows, cols, data_bm25 = [], [], []

        for t_idx, doc_tf in self.term_doc_tf.items():
            df = len(doc_tf)
            if df == 0:
                continue

            idf = np.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            self.idf[t_idx] = np.float32(idf)

            for d_idx, tf in doc_tf.items():
                dl = self.doc_lens_arr[d_idx]
                rows.append(d_idx)
                cols.append(t_idx)
                data_bm25.append(self._bm25(tf, dl, idf))

        self.matrix_bm25 = csr_matrix(
            (data_bm25, (rows, cols)),
            shape=(self.n_docs, self.n_terms),
        )
        self.matrix_bin = csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(self.n_docs, self.n_terms),
        )


    def _term_indices(self, query_terms):
        return [self.term_to_idx[t] for t in query_terms if t in self.term_to_idx]

    def _select_rows(self, candidate_ids):
        if candidate_ids is None:
            return slice(None), None
        cand_idx = [self.doc_to_idx[d] for d in candidate_ids if d in self.doc_to_idx]
        return cand_idx, cand_idx


    def get_scores(self, query_terms, candidate_ids=None, top_k=None):
        term_idxs = self._term_indices(query_terms)
        if not term_idxs:
            return [] if top_k is not None else {}

        row_sel, cand_idx = self._select_rows(candidate_ids)
        if candidate_ids is not None and not row_sel:
            return [] if top_k is not None else {}

        idf_sub = self.idf[term_idxs]

        Mbin = self.matrix_bin[row_sel, :][:, term_idxs]
        Mbm = self.matrix_bm25[row_sel, :][:, term_idxs]

        overlap = np.asarray(Mbin.dot(idf_sub)).ravel()
        bm25 = np.asarray(Mbm.sum(axis=1)).ravel()
        scores = (self.weights[0] * overlap) + (self.weights[1] * bm25)

        if top_k is not None:
            k = min(int(top_k), len(scores))
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top])]

            if cand_idx is not None:
                return [(self.doc_ids[cand_idx[i]], float(scores[i])) for i in top]
            return [(self.doc_ids[i], float(scores[i])) for i in top]

        nz = np.nonzero(scores > 0)[0]
        if cand_idx is not None:
            return {self.doc_ids[cand_idx[i]]: float(scores[i]) for i in nz}
        return {self.doc_ids[i]: float(scores[i]) for i in nz}
