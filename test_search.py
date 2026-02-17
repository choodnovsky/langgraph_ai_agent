#!/usr/bin/env python3
"""
Очистка тестового файла и переиндексация
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.settings import settings

# Пробуем оба варианта импорта
try:
    from src.components.indexer_core import DocumentWatcher
except ImportError:
    from src.indexer_core import DocumentWatcher

print("=" * 80)
print("🧹 Очистка и переиндексация")
print("=" * 80)

# Удаляем тестовый файл
test_file = Path(settings.FOLDER_PATH) / "test_indexer.txt"
if test_file.exists():
    test_file.unlink()
    print(f"✓ Удалён: {test_file.name}")
else:
    print(f"⚠️  Тестовый файл не найден")

# Создаём watcher
watcher = DocumentWatcher(settings)

# Инициализация
if not watcher.initialize():
    print("❌ Ошибка инициализации")
    sys.exit(1)

print(f"\nКоллекция: {settings.COLLECTION_NAME}")
print(f"Документов ДО очистки: {watcher.collection.count()}")

# Удаляем тестовый файл из ChromaDB
try:
    existing = watcher.collection.get(where={"filename": "test_indexer.txt"})
    if existing and existing['ids']:
        watcher.collection.delete(ids=existing['ids'])
        print(f"✓ Удалён из ChromaDB: test_indexer.txt")
    else:
        print(f"⚠️  test_indexer.txt не найден в ChromaDB")
except Exception as e:
    print(f"⚠️  Ошибка удаления из ChromaDB: {e}")

print(f"Документов ПОСЛЕ очистки: {watcher.collection.count()}")

# Тест поиска
print("\n🔍 Тест поиска 'Power BI дашборды':")
results = watcher.collection.query(
    query_texts=["Power BI дашборды"],
    n_results=3
)

if results['documents'] and results['documents'][0]:
    print(f"✅ Найдено документов: {len(results['documents'][0])}")
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
        filename = meta.get('filename', 'Unknown')
        chunk = meta.get('chunk_index', '?')
        total = meta.get('total_chunks', '?')
        preview = doc[:150]
        print(f"\n[{i}] {filename} (chunk {chunk}/{total})")
        print(f"    {preview}...")

        # Ищем упоминание Power BI
        if "Power BI" in doc or "power bi" in doc.lower():
            print(f"    💡 Содержит 'Power BI'!")
else:
    print("❌ Ничего не найдено")

print("\n" + "=" * 80)
print("✅ Очистка завершена")
print("=" * 80)