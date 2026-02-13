# src/components/process_txt_documents.py

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = DirectoryLoader(
    "wiki",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)

docs_list = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

doc_splits = text_splitter.split_documents(docs_list)

if __name__ == "__main__":
    if doc_splits:
        print("Chunks:", len(doc_splits))
        print("Sample:\n", doc_splits[0].page_content.strip())
        print("Metadata:", doc_splits[0].metadata)
    else:
        print("Нет загруженных документов")