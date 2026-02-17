# src/components/retriever_tool.py

from functools import lru_cache
from langchain.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings as ChromaSettings


# ============================================================================
# ChromaDB вариант (для txt файлов из watch-folder)
# ============================================================================

from functools import lru_cache

@lru_cache(maxsize=1)
def get_chroma_client():
    """Подключение к ChromaDB"""
    from src.settings import settings

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
    """Получить коллекцию из ChromaDB"""
    from src.settings import settings

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

        # Поиск - берём больше документов для лучшего покрытия
        results = collection.query(
            query_texts=[query],
            n_results=5  # Увеличили с 3 до 5
        )

        # DEBUG: показываем что нашли
        print(f"\n[DEBUG] Запрос: {query}")
        print(f"[DEBUG] Найдено документов: {len(results['documents'][0]) if results['documents'] else 0}")

        if not results['documents'] or not results['documents'][0]:
            return "Документы не найдены."

        # Форматирование
        documents = results['documents'][0]
        metadatas = results['metadatas'][0] if results['metadatas'] else []

        # DEBUG: показываем источники
        for i, meta in enumerate(metadatas):
            filename = meta.get('filename', 'Unknown') if meta else 'Unknown'
            preview = documents[i][:100] if i < len(documents) else ''
            print(f"[DEBUG] Документ {i+1}: {filename}")
            print(f"[DEBUG] Превью: {preview}...")

        formatted = []
        for doc, meta in zip(documents, metadatas):
            source = meta.get('filename', 'Unknown') if meta else 'Unknown'
            formatted.append(f"[Источник: {source}]\n{doc}")

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        return f"Ошибка поиска: {e}"



retriever_tool = retrieve_docs