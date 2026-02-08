from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = SemanticChunker(embeddings)

with open("./rag_article.txt", "r") as f:
    text = f.read()

docs = text_splitter.create_documents([text])

for i, doc in enumerate(docs, start=1):
    print('-' * 50)
    print(f"Chunk: {i}")
    print(doc.page_content)
