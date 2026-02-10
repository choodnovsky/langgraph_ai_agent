import os
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from chromadb import HttpClient

from src.custom_embeddings import CustomEmbeddings

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = "wiki_docs"

# Подключение к ChromaDB
client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# Embeddings
embeddings = CustomEmbeddings()

# VectorStore
vectorstore = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)


@tool
def retriever_tool(query: str) -> str:
    """Поиск по wiki (ChromaDB в Docker или локально)."""
    try:
        docs = vectorstore.similarity_search(query, k=5)

        if not docs:
            return "Документы не найдены."

        return "\n\n".join([doc.page_content for doc in docs])

    except Exception as e:
        return f"Ошибка при поиске: {str(e)}"