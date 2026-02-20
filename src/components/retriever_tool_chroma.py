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

    chroma_host      = os.getenv("CHROMA_HOST")
    chroma_port      = int(os.getenv("CHROMA_PORT"))
    collection_name  = os.getenv("COLLECTION_NAME")
    embeddings_model = os.getenv("EMBEDDINGS_MODEL")

    # Подключаемся к уже запущенному ChromaDB-серверу
    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

    # Та же модель эмбеддингов, что использует indexer.py
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)

    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    return vectorstore


@lru_cache(maxsize=1)
def get_retriever():
    """Получить retriever из ChromaDB (кэшируется)."""
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": 3})


@tool
def retrieve_docs(query: str) -> str:
    """Поиск и получение информации из документов.

    Ищет релевантную информацию в ChromaDB, которая содержит
    локальные текстовые документы из папки wiki.

    Args:
        query: Поисковый запрос (на русском или английском языке)

    Returns:
        Объединённый текст найденных документов
    """
    retriever = get_retriever()
    docs = retriever.invoke(query)
    result = "\n\n---\n\n".join([doc.page_content for doc in docs])
    return result


# Экспортируем инструмент (совместимо с graph_builder.py)
retriever_tool = retrieve_docs