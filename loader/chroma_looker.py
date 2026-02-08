"""
Просмотр содержимого ChromaDB через HTTP
- список коллекций
- размерность embedding
- первые 5 документов
"""

import chromadb
from src.settings import settings

# Инициализация HTTP-клиента
client = chromadb.HttpClient(
    host=settings.CHROMA_HOST,
    port=settings.CHROMA_PORT
)

print("=" * 80)
print("Список коллекций в ChromaDB")
print("=" * 80)

collections = client.list_collections()

if not collections:
    print("Коллекций не найдено")
    exit(0)

for col in collections:
    print(f"\nКоллекция: {col.name} (id: {col.id})")

    collection = client.get_collection(name=col.name)

    count = collection.count()
    print(f"Количество документов: {count}")

    if count == 0:
        print("Коллекция пустая")
        continue

    # ids возвращаются всегда, include их НЕ указываем
    data = collection.get(
        limit=5,
        include=["documents", "metadatas", "embeddings"]
    )

    ids = data.get("ids", [])
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])
    embeds = data.get("embeddings", [])

    # Размерность embedding
    if len(embeds) > 0:
        vector_dim = len(embeds[0])
        print(f"Размерность embedding: {vector_dim}")
    else:
        print("Размерность embedding: неизвестна")

    print("Первые документы:")

    for i in range(len(ids)):
        text_preview = docs[i][:100].replace("\n", " ")
        if len(docs[i]) > 100:
            text_preview += "..."
        print(f"{i+1}. ID: {ids[i]}")
        print(f"   Текст: {text_preview}")
        print(f"   Metadata: {metas[i]}")

print("\nПросмотр ChromaDB завершён")
print("=" * 80)