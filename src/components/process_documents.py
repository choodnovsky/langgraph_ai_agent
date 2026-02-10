from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from chromadb import HttpClient

from src.custom_embeddings import CustomEmbeddings


WIKI_PATH = Path("./wiki")
COLLECTION_NAME = "wiki_docs"


def load_txt_documents(path: Path):
    docs = []
    for file in path.glob("*.txt"):
        loader = TextLoader(str(file), encoding="utf-8")
        file_docs = loader.load()

        for doc in file_docs:
            doc.metadata["source"] = file.name

        docs.extend(file_docs)

    return docs


if __name__ == "__main__":
    docs = load_txt_documents(WIKI_PATH)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    doc_splits = splitter.split_documents(docs)

    embeddings = CustomEmbeddings()

    client = HttpClient(
        host="localhost",
        port=8000
    )

    vectordb = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

    vectordb.add_documents(doc_splits)