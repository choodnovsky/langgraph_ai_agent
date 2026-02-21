# graph/nodes/process_web_docs.py

from functools import lru_cache
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


@lru_cache(maxsize=1)
def get_web_documents():
    """Загрузить и разбить веб-документы.

    Вызывается только при первом использовании, результат кэшируется.
    Это предотвращает медленную инициализацию при импорте.
    """
    # URL блогов Лилиан Венг
    urls = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]


    # Загружаем документы с каждого URL
    docs = [WebBaseLoader(url).load() for url in urls]

    # Разворачиваем список списков в плоский список
    docs_list = [item for sublist in docs for item in sublist]

    # Унифицированные параметры с process_txt_docs.py
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    doc_splits = text_splitter.split_documents(docs_list)

    return doc_splits


# Для обратной совместимости (если кто-то импортирует doc_splits напрямую)
# НО это вызовет загрузку при импорте - не рекомендуется
# doc_splits = get_web_documents()