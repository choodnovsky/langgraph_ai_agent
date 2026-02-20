# src/components/retriever_tool_chroma.py

import os
from functools import lru_cache
from langchain.tools import tool


@lru_cache(maxsize=1)
def get_vectorstore():
    """Подключение к ChromaDB и возврат LangChain-обёртки над коллекцией.

    Инициализируется один раз при первом вызове, затем кэшируется.
    """
    import chromadb
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from src.settings import settings

    client = chromadb.HttpClient(
        host=settings.CHROMA_HOST,
        port=int(settings.CHROMA_PORT),
    )

    # Та же модель эмбеддингов, что использует indexer.py
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDINGS_MODEL)

    vectorstore = Chroma(
        client=client,
        collection_name=settings.COLLECTION_NAME,
        embedding_function=embeddings,
    )

    return vectorstore


@lru_cache(maxsize=1)
def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": 3})


@tool
def retrieve_docs(query: str) -> str:
    """Поиск и получение информации из документов.

    Ищет релевантную информацию в ChromaDB из локальных текстовых документов.

    Args:
        query: Поисковый запрос (на русском или английском языке)

    Returns:
        Объединённый текст найденных документов
    """
    docs = get_retriever().invoke(query)
    return "\n\n---\n\n".join([doc.page_content for doc in docs])


retriever_tool = retrieve_docs