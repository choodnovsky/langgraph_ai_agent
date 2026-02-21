# src/components/retriever_tool.py

from functools import lru_cache
from langchain.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def get_vectorstore():
    """Ленивая инициализация векторного хранилища.

    Загрузка документов и создание embeddings происходит только при первом вызове.
    Результат кэшируется для повторного использования.
    """

    # Импортируем только когда нужно
    from graph.nodes.process_web_docs import get_web_documents
    from graph.nodes.process_txt_docs import get_txt_documents

    # Получаем документы
    web_docs = get_web_documents()
    txt_docs = get_txt_documents()
    all_docs = web_docs + txt_docs

    # Создаем embeddings
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

    # Создаем векторное хранилище
    vectorstore = InMemoryVectorStore.from_documents(
        documents=all_docs,
        embedding=embeddings
    )

    return vectorstore


@lru_cache(maxsize=1)
def get_retriever():
    """Получить retriever (кэшируется)."""
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": 3})


@tool
def retrieve_docs(query: str) -> str:
    """Поиск и получение информации из документов.

    Ищет релевантную информацию в базе документов, которая включает:
    - Блоги Лилиан Венг (на английском)
    - Локальные текстовые документы (на русском/английском)

    Args:
        query: Поисковый запрос (на русском или английском языке)

    Returns:
        Объединенный текст найденных документов
    """
    # Retriever инициализируется только при первом вызове
    retriever = get_retriever()
    docs = retriever.invoke(query)

    # Объединяем содержимое найденных документов
    result = "\n\n---\n\n".join([doc.page_content for doc in docs])

    return result


# Экспортируем инструмент
retriever_tool = retrieve_docs