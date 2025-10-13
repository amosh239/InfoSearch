import re

_TOKEN_SPLIT_RE = re.compile(r"[\W_]+", re.UNICODE)

def normalize(text: str) -> str:
    return text.lower()

def tokenize(text: str):
    t = normalize(text)
    return [x for x in _TOKEN_SPLIT_RE.split(t) if x]
