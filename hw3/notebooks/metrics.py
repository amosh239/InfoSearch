import numpy as np

def mrr(ranked_results, relevant_ids, k=10):
    for rank, doc_id in enumerate(ranked_results[:k]):
        if doc_id in relevant_ids:
            return 1.0 / (rank + 1)
    return 0.0

def ndcg(ranked_results, relevant_ids, k=10):
    dcg = 0.0
    idcg = 0.0
    
    # DCG: real score
    for i, doc_id in enumerate(ranked_results[:k]):
        if doc_id in relevant_ids:
            dcg += 1.0 / np.log2(i + 2)
            
    # IDCG: ideal score
    num_relevant = min(len(relevant_ids), k)
    for i in range(num_relevant):
        idcg += 1.0 / np.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0