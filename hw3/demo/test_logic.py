"""Lightweight logic regression tests for the minisearch demo.

How to run (from repo root):
  python -m demo.test_logic

Or, if you prefer running as a script:
  python demo/test_logic.py

The file is intentionally dependency-free (no pytest/unittest required).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Set


# Ensure project root is on sys.path (works both for -m and direct execution).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisearch.index import PositionalIndex
from minisearch.search import Searcher

try:
    from .sample_docs import SAMPLE_DOCS  # when demo is a package
except Exception:
    from sample_docs import SAMPLE_DOCS  # when executed as a script


def _build_searcher() -> Searcher:
    idx = PositionalIndex()
    for i, d in enumerate(SAMPLE_DOCS):
        idx.add_document(i, d)
    idx.commit()
    return Searcher(idx)


def _as_set(xs: Iterable[int]) -> Set[int]:
    return set(xs)


def _fmt_ids(ids: Iterable[int]) -> str:
    return "[" + ", ".join(map(str, sorted(ids))) + "]"


def run_tests() -> None:
    sr = _build_searcher()

    failures: List[str] = []

    def assert_search(query: str, expected_ids: Iterable[int], comment: str) -> None:
        got_ids = [doc_id for doc_id, _ in sr.search(query)]
        got, expected = _as_set(got_ids), _as_set(expected_ids)
        if got != expected:
            failures.append(
                f"FAIL | {comment}\n"
                f"  Q: {query!r}\n"
                f"  got:      {_fmt_ids(got)}\n"
                f"  expected: {_fmt_ids(expected)}"
            )
        else:
            print(f"PASS | {comment} | Q: {query!r} -> {_fmt_ids(expected)}")

    def assert_no_crash(query: str, comment: str) -> None:
        try:
            _ = sr.search(query)
            print(f"PASS | {comment} | Q: {query!r}")
        except Exception as e:
            failures.append(
                f"FAIL | {comment}\n"
                f"  Q: {query!r}\n"
                f"  exception: {type(e).__name__}: {e}"
            )

    n_docs = len(SAMPLE_DOCS)
    all_docs = set(range(n_docs))

    print("--- Running minisearch logic tests ---")
    print(f"Docs indexed: {n_docs}")

    # ---------------------------------------------------------------------
    # Phrase / proximity semantics
    # ---------------------------------------------------------------------
    assert_search('"летний дождь"', [0], "Phrase: exact")
    assert_search('"летний дождь прошёл"', [0], "Phrase: longer phrase")
    assert_search('"молоко хлеб"', [3], "Phrase: punctuation-insensitive adjacency")
    assert_search('"кофе кофе"', [6], "Phrase: repeated term")

    # Negative phrase tests
    assert_search('"белый медведь"', [], "Phrase: distant words must not match")
    assert_search('"белый дом"', [], "Phrase: must not match across fields")
    assert_search('"дождь летний"', [], "Phrase: wrong order must not match")

    # NEAR/k
    assert_search('кот NEAR/4 мышь', [], "NEAR: too strict")
    assert_search('кот NEAR/6 мышь', [9], "NEAR: loose enough")
    assert_search('мышь NEAR/6 кот', [9], "NEAR: symmetry")

    # Distant terms: AND vs NEAR
    assert_search('трамваи AND дождь', [5], "AND: distant terms still match")
    assert_search('трамваи NEAR/5 дождь', [], "NEAR: distant terms must not match")

    # ---------------------------------------------------------------------
    # Boolean logic: precedence, parentheses, NOT
    # ---------------------------------------------------------------------
    assert_search('кот OR дом AND кофе', [1, 6, 9, 10], "Precedence: AND binds tighter than OR")
    assert_search('(кот OR дом) AND кофе', [6, 10], "Parentheses override precedence")

    assert_search('NOT кот AND кофе', [4, 10, 11], "NOT + AND")
    assert_search('NOT (кот AND кофе)', sorted(all_docs - {6}), "NOT scope with parentheses")

    # De Morgan consistency
    assert_search('NOT (город OR кофе)', [1, 3, 7, 8, 9], "De Morgan: NOT(OR)")
    assert_search('(NOT город) AND (NOT кофе)', [1, 3, 7, 8, 9], "De Morgan: AND of NOTs")

    # ---------------------------------------------------------------------
    # Field filters
    # ---------------------------------------------------------------------
    assert_search('title:кофе', [4, 6, 10], "Field: title")
    assert_search('tags:город', [0, 2, 5], "Field: tags")
    assert_search('body:город', [5, 12], "Field: body")

    assert_search('tags:кофе', [4, 6, 10, 11], "Field: tags (coffee only)")
    assert_search('body:кофе', [4, 6, 10], "Field: body (coffee)")

    # ---------------------------------------------------------------------
    # Wildcard / fuzzy
    # ---------------------------------------------------------------------
    assert_search('трам*', [2, 5], "Wildcard: prefix *")
    assert_search('коф?', [4, 6, 10, 11], "Wildcard: single-char ?")
    assert_search('кот~1', [1, 6, 9], "Fuzzy: edit distance <= 1")

    # ---------------------------------------------------------------------
    # Robustness (should not throw)
    # ---------------------------------------------------------------------
    assert_search('', [], "Empty query")
    assert_no_crash('(кот AND кофе', "Unbalanced parentheses should not crash")
    assert_no_crash('"летний дождь', "Unclosed quote should not crash")

    print("--- Summary ---")
    if failures:
        print(f"FAILED: {len(failures)}")
        for f in failures:
            print("\n" + f)
        raise SystemExit(1)

    print("ALL TESTS PASSED")


if __name__ == '__main__':
    run_tests()
