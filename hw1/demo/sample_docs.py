SAMPLE_DOCS = [
    {
        "title": "Введение в информационный поиск",
        "body": "Обратный индекс хранит для каждого термина список документов и позиций. Фразовый поиск требует координатного индекса.",
        "tags": "ir, индекс, координатный, фразы"
    },
    {
        "title": "Boolean retrieval model",
        "body": "Extended boolean queries support operators AND, OR, NOT and proximity such as NEAR/3.",
        "tags": "ir, boolean, near"
    },
    {
        "title": "Полевая индексация",
        "body": "Поля документа: title, body, tags. Запросы вида title:поиск AND body:\"обратный индекс\".",
        "tags": "fields, title, body, tags"
    },
    {
        "title": "Fuzzy & wildcard",
        "body": "Поддерживаются маски типа data*, gr?ph, а также неточное совпадение term~1.",
        "tags": "wildcard, fuzzy"
    },

    {
        "title": "Ранжирование и TF-IDF",
        "body": "TF IDF и BM25 улучшают качество выдачи по сравнению с простым подсчетом совпадений.",
        "tags": "tf, idf, bm25, ranking"
    },
    {
        "title": "BM25 baseline",
        "body": "BM25 uses term frequency, inverse document frequency and field length normalization.",
        "tags": "bm25, ranking, baseline"
    },
    {
        "title": "Поисковые подсказки и синонимы",
        "body": "Подсказки помогают пользователю формулировать запрос. Синонимы можно расширять на этапе парсинга или индекса.",
        "tags": "suggest, synonyms, query"
    },
    {
        "title": "Позиционный индекс: фразы и порядок",
        "body": "Позиционный индекс позволяет искать точные фразы и учитывать порядок слов в тексте.",
        "tags": "positional, phrase, order"
    },
    {
        "title": "K-gram индекс и wildcard",
        "body": "K-gram индекс ускоряет расширение масок. Примеры слов: airflow, dataflow, overflow, outflow.",
        "tags": "kgram, wildcard, *flow"
    },
    {
        "title": "Fuzzy примеры на английском",
        "body": "Common misspellings: retrival, retreival, retreval should match retrieval with term~1 or term~2.",
        "tags": "fuzzy, levenshtein, english"
    },
    {
        "title": "Префиксные деревья (trie) и автодополнение",
        "body": "Префиксное дерево удобно для автодополнения по префиксу: data, database, dataset, dataframe.",
        "tags": "trie, prefix, autocomplete"
    },
    {
        "title": "Лемматизация и стемминг",
        "body": "Лемматизация и стемминг приводят формы слов: поиск, поиска, поисковый, поисковая. Пока используем wildcard индек*.",
        "tags": "nlp, stemming, lemmatization"
    },
    {
        "title": "Полевые запросы в продуктах",
        "body": "Примеры: title:\"быстрый поиск\" OR body:индекс; tags:ir AND NOT tags:db.",
        "tags": "fields, title, body, tags, boolean"
    },
    {
        "title": "Proximity search playground",
        "body": "Boolean NEAR/2 queries should match when terms are within the given distance.",
        "tags": "near, proximity, boolean"
    },
    {
        "title": "Стоп-слова и нормализация",
        "body": "Стоп слова фильтруются редко в современном поиске; простая нормализация: lower-case и разбиение по знакам.",
        "tags": "stopwords, normalize"
    },
    {
        "title": "Русские опечатки",
        "body": "Опечатки вроде индеск и поик должны находиться фаззи-поиском: индекс~1, поиск~1.",
        "tags": "fuzzy, russian, typos"
    },
    {
        "title": "Реляционные БД vs полнотекстовый поиск",
        "body": "PostgreSQL может использовать tsvector, но специализированные движки дают лучший скор и масштабирование.",
        "tags": "postgres, tsvector, fulltext"
    },
    {
        "title": "Сжатие списков постингов",
        "body": "VarInt, Simple-9 и PForDelta уменьшают размер обратного индекса и ускоряют чтение.",
        "tags": "compression, varint, pfor"
    },
    {
        "title": "Морфология: индекс/индекса/индексация",
        "body": "Демонстрация форм: индекс, индекса, индексам, индексу, индексация, индексный. Хорошо тестировать индек*.",
        "tags": "morphology, russian, wildcard"
    },
    {
        "title": "NEAR/0 как фраза",
        "body": "Обратный индекс встречается рядом: обратный индекс. Это эквивалент фразового совпадения или NEAR/0.",
        "tags": "near, phrase, exact"
    }
]
