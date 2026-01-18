import os
import random
import pandas as pd
import ir_datasets
from tqdm import tqdm
from minisearch.index import PositionalIndex

DATA_DIR = "data"

def download_and_sample(force_reload=False, n_queries=500, n_docs=10000):
    os.makedirs(DATA_DIR, exist_ok=True)
    paths = {
        "docs": f"{DATA_DIR}/docs.csv",
        "queries": f"{DATA_DIR}/queries.csv",
        "qrels": f"{DATA_DIR}/qrels.csv"
    }

    if all(os.path.exists(p) for p in paths.values()) and not force_reload:
        print("Dataset already exists. Skipping download.")
        return

    print("Loading MS MARCO dataset stream...")
    dataset = ir_datasets.load("msmarco-document/train")

    print(f"Sampling {n_queries} queries...")
    all_queries = list(dataset.queries_iter())
    queries = random.sample(all_queries, n_queries)
    q_ids = {q.query_id for q in queries}

    print("Filtering qrels...")
    qrels = [r for r in dataset.qrels_iter() if r.query_id in q_ids]
    rel_doc_ids = {r.doc_id for r in qrels}

    print(f"Sampling {n_docs} documents (relevant + noise)...")
    docs = []
    for doc in tqdm(dataset.docs_iter(), desc="Scanning corpus"):
        is_relevant = doc.doc_id in rel_doc_ids
        
        if is_relevant or len(docs) < n_docs:
            docs.append({"doc_id": doc.doc_id, "title": doc.title, "body": doc.body})
            if is_relevant:
                rel_doc_ids.remove(doc.doc_id)
        
        if len(docs) >= n_docs and not rel_doc_ids:
            break

    print("Saving to CSV...")
    pd.DataFrame(docs).to_csv(paths["docs"], index=False)
    pd.DataFrame(queries).to_csv(paths["queries"], index=False)
    pd.DataFrame(qrels).to_csv(paths["qrels"], index=False)
    print("Done.")

def load_data():
    docs = pd.read_csv(f"{DATA_DIR}/docs.csv").fillna("")
    queries = pd.read_csv(f"{DATA_DIR}/queries.csv")
    
    qrels_df = pd.read_csv(f"{DATA_DIR}/qrels.csv")
    qrels = qrels_df.groupby("query_id")["doc_id"].apply(set).to_dict()
    
    return docs, queries, qrels

def build_index(docs_df):
    print("Indexing...")
    index = PositionalIndex()
    int_to_id = {}

    for i, row in tqdm(docs_df.iterrows(), total=len(docs_df)):
        index.add_document(i, {"title": str(row["title"]), "body": str(row["body"])})
        int_to_id[i] = row["doc_id"]
    
    index.commit()
    return index, int_to_id