import re
from dataclasses import dataclass
from typing import List, Optional
from .analysis import normalize, tokenize

class Node: pass

@dataclass
class TermNode(Node):
    term: str
    field: Optional[str] = None
    wildcard: bool = False
    fuzzy: int = 0

@dataclass
class PhraseNode(Node):
    terms: List[str]
    field: Optional[str] = None

@dataclass
class NearNode(Node):
    left: Node; right: Node; k: int = 1

@dataclass
class AndNode(Node):
    left: Node; right: Node

@dataclass
class OrNode(Node):
    left: Node; right: Node

@dataclass
class NotNode(Node):
    child: Node

def parse_query(query: str) -> Node:
    tokens = re.findall(r'\(|\)|"[^"]*"|[^\s()]+', query)
    output: List[Node] = []
    ops: List[str] = []
    prec = {'NOT': 3, '!': 3, 'AND': 1, '&&': 1, 'OR': 0, '||': 0}

    def apply_op(op):
        if op in ('NOT', '!'):
            if output: output.append(NotNode(output.pop()))
        elif op in ('AND', '&&', 'OR', '||'):
            if len(output) < 2: return
            b, a = output.pop(), output.pop()
            output.append(AndNode(a, b) if op in ('AND', '&&') else OrNode(a, b))
        elif op.upper().startswith('NEAR/'):
            if len(output) < 2: return
            k = int(op.split('/')[1])
            b, a = output.pop(), output.pop()
            output.append(NearNode(a, b, k))

    for t in tokens:
        up = t.upper()
        if up in prec:
            while ops and ops[-1] != '(' and prec.get(ops[-1], -1) >= prec[up]:
                apply_op(ops.pop())
            ops.append(up)
        elif up.startswith('NEAR/'):
             ops.append(up)
        elif t == '(':
            ops.append(t)
        elif t == ')':
            while ops and ops[-1] != '(':
                apply_op(ops.pop())
            if ops: ops.pop()
        else:
            field, body = None, t
            if ':' in t and not t.startswith('"'):
                field, body = t.split(':', 1)
            
            if body.startswith('"'):
                clean_terms = [normalize(w) for w in tokenize(body.strip('"'))]
                output.append(PhraseNode(clean_terms, field))
            else:
                wild = '*' in body or '?' in body
                fuz = 0
                if '~' in body and not wild:
                    try: body, fuz = body.split('~', 1); fuz = int(fuz)
                    except: fuz = 1
                output.append(TermNode(normalize(body), field, wild, fuz))

    while ops:
        op = ops.pop()
        if op != '(': apply_op(op)

    while len(output) > 1:
        b = output.pop()
        a = output.pop()
        output.append(AndNode(a, b))

    return output[-1] if output else TermNode("")