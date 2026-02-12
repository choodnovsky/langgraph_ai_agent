# src/components/retriever_tool.py

from langchain.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from src.components.process_web_documents import doc_splits as web_doc_splits
from src.components.process_txt_documents import doc_splits as txt_doc_splits


# --- 1. Объединяем документы ---
all_docs = web_doc_splits + txt_doc_splits


# --- 2. Локальные эмбеддинги ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --- 3. Векторное хранилище ---
vectorstore = InMemoryVectorStore.from_documents(
    documents=all_docs,
    embedding=embeddings
)


retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# --- 4. Tool ---
@tool
def retrieve_documents(query: str) -> str:
    """Поиск релевантных фрагментов документов."""
    docs = retriever.invoke(query)

    if not docs:
        return "Релевантные документы не найдены."

    return "\n\n".join([doc.page_content for doc in docs])


retriever_tool = retrieve_documents