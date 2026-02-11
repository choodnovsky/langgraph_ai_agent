# scripts/drop_collection.py

import chromadb
from src.settings import settings

print("=" * 70)
print("ПОЛНОЕ УДАЛЕНИЕ КОЛЛЕКЦИИ CHROMADB")
print("=" * 70)

client = chromadb.HttpClient(
    host=settings.CHROMA_HOST,
    port=settings.CHROMA_PORT,
)

client.delete_collection(settings.COLLECTION_NAME)

print(f"✓ Коллекция '{settings.COLLECTION_NAME}' удалена")