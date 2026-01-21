from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisearch.index import PositionalIndex
from minisearch.search import Searcher

try:
    from .sample_docs import SAMPLE_DOCS
except Exception:
    from sample_docs import SAMPLE_DOCS


def _build_searcher() -> Searcher:
    idx = PositionalIndex()
    for doc_id, doc in enumerate(SAMPLE_DOCS):
        idx.add_document(doc_id, doc)
    if hasattr(idx, "commit"):
        idx.commit()
    return Searcher(idx)


def _as_set(xs: Iterable[int]) -> Set[int]:
    return set(xs)


def _fmt_ids(ids: Iterable[int]) -> str:
    return "[" + ", ".join(map(str, sorted(ids))) + "]"


def run_tests() -> None:
    sr = _build_searcher()

    failures: List[str] = []
    n_docs = len(SAMPLE_DOCS)
    all_docs = set(range(n_docs))

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

    print("--- Running boolean logic tests ---")
    print(f"Docs indexed: {n_docs}")

    # Приоритеты: AND сильнее OR
    assert_search("кот OR дом AND кофе", [1, 6, 9, 10], "Precedence: AND binds tighter than OR")
    assert_search("(кот OR дом) AND кофе", [6, 10], "Parentheses override precedence")

    # Базовые OR / AND
    assert_search("кот OR дом OR кофе", [1, 3, 4, 6, 8, 9, 10, 11], "OR: union")
    assert_search("кот AND дом", [1], "AND: intersection")

    # NOT и скобки
    assert_search("NOT кот AND кофе", [4, 10, 11], "NOT + AND")
    assert_search("NOT (кот AND кофе)", sorted(all_docs - {6}), "NOT scope with parentheses")

    # Де Морган (на консистентность)
    assert_search("NOT (город OR кофе)", [1, 3, 7, 8, 9], "De Morgan: NOT(OR)")
    assert_search("(NOT город) AND (NOT кофе)", [1, 3, 7, 8, 9], "De Morgan: AND of NOTs")

    print("--- Summary ---")
    if failures:
        print(f"FAILED: {len(failures)}")
        for f in failures:
            print("\n" + f)
        raise SystemExit(1)

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    run_tests()
