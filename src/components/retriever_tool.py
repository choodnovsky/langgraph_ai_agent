from langchain.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from src.components.process_web_documents import doc_splits as web_doc_splits
from src.components.process_txt_documents import doc_splits as txt_doc_splits
from langgraph.graph import MessagesState

# Объединяем документы
all_docs = web_doc_splits + txt_doc_splits

# Эмбеддинги
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

# Векторное хранилище
vectorstore = InMemoryVectorStore.from_documents(
    documents=all_docs,
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

@tool
def retrieve_docs(query: str) -> str:
    """Поиск и получение информации из корпаративной wiki. Запрос должен быть на русском языке."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


retriever_tool = retrieve_docs
