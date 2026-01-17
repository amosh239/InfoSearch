import sys
import os

from minisearch.index import PositionalIndex
from minisearch.search import Searcher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from demo.sample_docs import SAMPLE_DOCS

def run_tests():
    # 1. Инициализация
    print("--- Indexing Docs ---")
    idx = PositionalIndex()
    for i, d in enumerate(SAMPLE_DOCS):
        idx.add_document(i, d)
    idx.commit()
    
    sr = Searcher(idx)

    # Хелпер для запуска и проверки
    def assert_search(query, expected_ids, comment):
        results = [id for id, _ in sr.search(query)]
        # Сортируем для сравнения множеств
        match = sorted(results) == sorted(expected_ids)
        status = "✅ PASS" if match else f"❌ FAIL (Got {results}, Expected {expected_ids})"
        print(f"{status} | Q: '{query}' | {comment}")

    print("\n--- Testing Logic ---")

    # ТЕСТ 1: Обычная фраза
    # Док 0: "Летний дождь..." -> Должен найти
    assert_search('"летний дождь"', [0], "Exact phrase match")

    # ТЕСТ 2: Фраза с разрывом (Negative Test)c
    # Док 7: "...Белый... медведь..." (слова есть, но не рядом) -> Не должен найти
    # Если вернет [7], значит работает как старый AND, а не Phrase
    assert_search('"белый медведь"', [], "Broken phrase should not match")

    # ТЕСТ 3: Фраза в разных полях (Negative Test)
    # Док 8: Title="Белый", Body="Дом..." -> Не должен найти
    assert_search('"белый дом"', [], "Cross-field phrase should fail")

    # ТЕСТ 4: Порядок слов (Negative Test)
    # Док 0: "Летний дождь" -> ищем "дождь летний" -> Не должен найти
    assert_search('"дождь летний"', [], "Wrong order phrase")

    # ТЕСТ 5: NEAR (близость)
    # Док 9: "Кот долго ждал и поймал мышь"
    # "Кот"(0) ... "мышь"(5). Дистанция 5.
    
    # NEAR/1 (слишком мало) -> Пусто
    assert_search('кот NEAR/1 мышь', [], "Too strict NEAR")
    
    # NEAR/6 (достаточно) -> Док 9
    assert_search('кот NEAR/6 мышь', [9], "Loose NEAR")

    # ТЕСТ 6: Сложный запрос (Комбинация)
    # Док 5 содержит "трамваи" и "дождь", но далеко друг от друга.
    # AND должен найти, NEAR/5 не должен.
    assert_search('трамваи AND дождь', [5], "AND operator (distant terms)")
    assert_search('трамваи NEAR/5 дождь', [], "NEAR operator (distant terms)")

if __name__ == "__main__":
    run_tests()