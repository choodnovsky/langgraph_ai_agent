# src/components/retriever_tool.py

from functools import lru_cache
from langchain.tools import tool
import chromadb
from chromadb.config import Settings as ChromaSettings


@lru_cache(maxsize=1)
def get_chroma_client():
    """Подключение к ChromaDB"""
    from src2.settings import settings

    try:
        client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=int(settings.CHROMA_PORT),
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        client.heartbeat()
        print(f"✓ ChromaDB: {settings.chroma_url}")
        return client
    except Exception as e:
        print(f"✗ Ошибка ChromaDB: {e}")
        raise


@lru_cache(maxsize=1)
def get_chroma_collection():
    """Получить коллекцию"""
    from src2.settings import settings

    client = get_chroma_client()

    try:
        collection = client.get_collection(name=settings.COLLECTION_NAME)
        count = collection.count()
        print(f"✓ Коллекция '{settings.COLLECTION_NAME}', документов: {count}")
        return collection
    except Exception as e:
        print(f"✗ Коллекция не найдена: {e}")
        print("  Запустите индексатор!")
        raise


@tool
def retrieve_docs(query: str) -> str:
    """Поиск в ChromaDB.

    Args:
        query: Поисковый запрос

    Returns:
        Найденные документы
    """
    try:
        collection = get_chroma_collection()

        # Поиск
        results = collection.query(
            query_texts=[query],
            n_results=3
        )

        if not results['documents'] or not results['documents'][0]:
            return "Документы не найдены."

        # Форматирование
        documents = results['documents'][0]
        metadatas = results['metadatas'][0] if results['metadatas'] else []

        formatted = []
        for doc, meta in zip(documents, metadatas):
            source = meta.get('filename', 'Unknown') if meta else 'Unknown'
            formatted.append(f"[Источник: {source}]\n{doc}")

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        return f"Ошибка поиска: {e}"


# Экспорт
retriever_tool = retrieve_docs