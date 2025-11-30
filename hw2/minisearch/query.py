from dataclasses import dataclass
from typing import List, Optional
import re
from .analysis import normalize, tokenize

class Node: ...

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
    left: Node
    right: Node
    k: int = 1

@dataclass
class AndNode(Node):
    left: Node
    right: Node

@dataclass
class OrNode(Node):
    left: Node
    right: Node

@dataclass
class NotNode(Node):
    child: Node

_OPS = {"AND", "OR", "NOT", "&&", "||", "!"}
_NEAR_RE = re.compile(r"NEAR\/(\d+)$", re.IGNORECASE)
_FIELD_TERM_RE = re.compile(r"^(?P<field>[a-zA-Z]\w*):(?P<body>.+)$")

def _lex(query: str) -> List[str]:
    qs = query.strip()
    tokens: List[str] = []
    i = 0
    while i < len(qs):
        c = qs[i]
        if c.isspace():
            i += 1; continue
        if c in '()':
            tokens.append(c); i += 1; continue
        if c == '"':
            j = i + 1; buf = []
            while j < len(qs) and qs[j] != '"':
                buf.append(qs[j]); j += 1
            if j >= len(qs): j = len(qs)
            tokens.append('"' + ''.join(buf) + '"')
            i = j + 1 if j < len(qs) else j
            continue
        j = i
        while j < len(qs) and not qs[j].isspace() and qs[j] not in '()':
            j += 1
        tokens.append(qs[i:j]); i = j
    return tokens

def parse_query(raw: str) -> Node:
    tks = _lex(raw)

    def prec(op: str) -> int:
        if op in ("NOT", "!"): return 3
        if _NEAR_RE.match(op): return 2
        if op in ("AND", "&&"): return 1
        if op in ("OR", "||"): return 0
        return -1

    output: List[Node] = []
    ops: List[str] = []

    def apply_op(op: str) -> None:
        if op in ("NOT", "!"):
            if not output: return
            a = output.pop(); output.append(NotNode(a)); return
        m = _NEAR_RE.match(op)
        if m:
            k = int(m.group(1))
            b = output.pop(); a = output.pop()
            output.append(NearNode(a, b, k)); return
        if op in ("AND", "&&"):
            b = output.pop(); a = output.pop()
            output.append(AndNode(a, b)); return
        if op in ("OR", "||"):
            b = output.pop(); a = output.pop()
            output.append(OrNode(a, b)); return

    def push_term(token: str) -> None:
        m = _FIELD_TERM_RE.match(token)
        field: Optional[str] = None
        body = token
        if m:
            field = m.group('field')
            body = m.group('body')
        if body.startswith('"') and body.endswith('"'):
            phrase = body.strip('"')
            terms = [normalize(t) for t in tokenize(phrase)]
            output.append(PhraseNode(terms, field)); return
        wildcard = ('*' in body) or ('?' in body)
        fuzzy = 0
        fm = re.search(r"~(\d?)$", body)
        if fm:
            body = body[:fm.start()]
            fuzzy = int(fm.group(1)) if fm.group(1) else 1
        term = normalize(body)
        output.append(TermNode(term=term, field=field, wildcard=wildcard, fuzzy=fuzzy))

    i = 0
    while i < len(tks):
        tok = tks[i]; up = tok.upper()
        if tok == '(':
            ops.append(tok); i += 1; continue
        if tok == ')':
            while ops and ops[-1] != '(':
                apply_op(ops.pop())
            if ops and ops[-1] == '(':
                ops.pop()
            i += 1; continue
        if up in _OPS or _NEAR_RE.match(up):
            if up in ('NOT', '!'):
                ops.append(up)
            elif _NEAR_RE.match(up):
                while ops and prec(ops[-1]) > prec(up):
                    apply_op(ops.pop())
                ops.append(up)
            elif up in ('AND', '&&', 'OR', '||'):
                while ops and prec(ops[-1]) >= prec(up):
                    apply_op(ops.pop())
                ops.append(up)
            i += 1; continue
        push_term(tok); i += 1

    while ops:
        op = ops.pop()
        if op in ('(', ')'): continue
        apply_op(op)

    return output[-1] if output else TermNode(term="")
