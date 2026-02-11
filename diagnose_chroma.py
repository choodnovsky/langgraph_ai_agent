#!/usr/bin/env python3
"""
Диагностика качества векторной базы ChromaDB
"""

import chromadb
from chromadb.config import Settings
from src.custom_embeddings import CustomEmbeddings
from src.settings import settings
from langchain_chroma import Chroma
import numpy as np


def normalize_chroma_list(x):
    """
    Приводит ответы Chroma к List[T]
    Поддерживает:
    - List[T]
    - List[List[T]]
    - numpy.ndarray
    """
    if x is None:
        return []

    if isinstance(x, np.ndarray):
        return x.tolist()

    if isinstance(x, list):
        if len(x) > 0 and isinstance(x[0], list):
            return x[0]
        return x

    raise TypeError(f"Unexpected Chroma format: {type(x)}")


def diagnose_chromadb():
    print("=" * 70)
    print("ДИАГНОСТИКА CHROMADB")
    print("=" * 70)

    client = chromadb.HttpClient(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=False,
        ),
    )

    collection = client.get_collection(name=settings.COLLECTION_NAME)

    count = collection.count()
    print(f"\n[INFO] Всего документов в коллекции: {count}")

    if count == 0:
        print("[ERROR] База пустая!")
        return

    raw = collection.get(
        include=["documents", "metadatas", "embeddings"]
    )

    documents = normalize_chroma_list(raw.get("documents"))
    metadatas = normalize_chroma_list(raw.get("metadatas"))
    embeddings = normalize_chroma_list(raw.get("embeddings"))

    print("\n[INFO] Примеры документов:")
    print("-" * 70)

    max_examples = min(5, len(documents))

    for i in range(max_examples):
        doc = documents[i]
        meta = metadatas[i] if i < len(metadatas) else {}

        print(f"\nДокумент {i + 1}:")
        print(f"  Источник: {meta.get('source', 'N/A')}")
        print(f"  Длина: {len(doc)} символов")
        print(f"  Превью: {doc[:200]}...")

        if embeddings is not None and len(embeddings) > i:
            print(f"  Размерность эмбеддинга: {len(embeddings[i])}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ТЕСТОВЫЕ ПОИСКИ")
    print("=" * 70)

    embedding_model = CustomEmbeddings()

    vectorstore = Chroma(
        client=client,
        collection_name=settings.COLLECTION_NAME,
        embedding_function=embedding_model,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    test_queries = [
        "что это за компания",
        "какие услуги предоставляет",
        "контактная информация",
        "о чем этот документ",
    ]

    for query in test_queries:
        print(f"\n[TEST] Запрос: '{query}'")
        docs = retriever.invoke(query)

        if not docs:
            print("  [WARN] Ничего не найдено")
            continue

        print(f"  [OK] Найдено: {len(docs)} документов")
        for i, doc in enumerate(docs[:2], start=1):
            print(f"    Док {i}: {doc.page_content[:150]}...")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("АНАЛИЗ МЕТАДАННЫХ")
    print("=" * 70)

    sources = {
        meta.get("source")
        for meta in metadatas
        if meta.get("source")
    }

    print(f"\n[INFO] Уникальных источников: {len(sources)}")
    for source in sorted(sources):
        print(f"  - {source}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("АНАЛИЗ РАЗМЕРА ЧАНКОВ")
    print("=" * 70)

    lengths = [len(doc) for doc in documents]

    print(f"\n[INFO] Статистика длины документов:")
    print(f"  Минимум: {min(lengths)}")
    print(f"  Максимум: {max(lengths)}")
    print(f"  Средняя: {sum(lengths) / len(lengths):.0f}")

    small = sum(x < 100 for x in lengths)
    medium = sum(100 <= x < 500 for x in lengths)
    large = sum(500 <= x < 1000 for x in lengths)
    xlarge = sum(x >= 1000 for x in lengths)

    print("\n[INFO] Распределение:")
    print(f"  < 100: {small} ({small / len(lengths) * 100:.1f}%)")
    print(f"  100–500: {medium} ({medium / len(lengths) * 100:.1f}%)")
    print(f"  500–1000: {large} ({large / len(lengths) * 100:.1f}%)")
    print(f"  > 1000: {xlarge} ({xlarge / len(lengths) * 100:.1f}%)")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("РЕКОМЕНДАЦИИ")
    print("=" * 70)

    avg_len = sum(lengths) / len(lengths)

    if avg_len < 150:
        print("[WARN] Чанки слишком маленькие")
        print("       Рекомендация: chunk_size 500–800")
    elif avg_len > 1500:
        print("[WARN] Чанки слишком большие")
        print("       Рекомендация: chunk_size 500–800")
    else:
        print("[OK] Размер чанков в норме")

    if count < 20:
        print("\n[WARN] Мало документов для устойчивого RAG")
        print("       Рекомендация: увеличить корпус")


if __name__ == "__main__":
    diagnose_chromadb()