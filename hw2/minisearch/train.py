import json
from sklearn.linear_model import LinearRegression
from demo.sample_docs import SAMPLE_DOCS
from minisearch.index import PositionalIndex
from minisearch.ranking import FeatureExtractor

idx = PositionalIndex()
for i, doc in enumerate(SAMPLE_DOCS):
    idx.add_document(i, doc)
idx.commit()

fe = FeatureExtractor(idx)
with open('minisearch/data.json', 'r') as f:
    data = json.load(f)

X = [fe.get_features(item['doc_id'], item['query'].lower().split()) for item in data]
y = [item['label'] for item in data]

model = LinearRegression(fit_intercept=False, positive=True).fit(X, y)
print(f"Learned weights: {model.coef_}")

with open('minisearch/weights.json', 'w') as f:
    json.dump(list(model.coef_), f)