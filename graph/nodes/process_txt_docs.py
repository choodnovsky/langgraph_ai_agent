# graph/nodes/process_txt_docs.py

from functools import lru_cache
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


@lru_cache(maxsize=1)
def get_txt_documents():
    """Загрузить и разбить текстовые документы.

    Вызывается только при первом использовании, результат кэшируется.
    Это предотвращает медленную инициализацию при импорте.
    """
    # Используем относительный путь от корня проекта
    project_root = Path(__file__).parent.parent.parent
    wiki_path = project_root / "wiki"

    # Проверяем существование директории
    if not wiki_path.exists():
        return []


    loader = DirectoryLoader(
        str(wiki_path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    docs_list = loader.load()
    # Унифицированные параметры сплиттера (как в web_documents)
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
# doc_splits = get_txt_documents()