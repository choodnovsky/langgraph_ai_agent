#!/usr/bin/env python3
"""
Тестовый скрипт для проверки подключения к ChromaDB и поиска документов
"""
import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

from src2.components.retriever_tool import get_chroma_client, get_chroma_collection


def main():
    print("=" * 80)
    print("🔍 Тестирование подключения к ChromaDB")
    print("=" * 80)
    
    try:
        # Подключение к ChromaDB
        print("\n1. Подключение к ChromaDB...")
        client = get_chroma_client()
        print("   ✓ Подключение установлено")
        
        # Получение коллекции
        print("\n2. Получение коллекции...")
        collection = get_chroma_collection()
        count = collection.count()
        print(f"   ✓ Коллекция найдена, документов: {count}")
        
        if count == 0:
            print("\n⚠️  Коллекция пуста. Добавьте документы в папку watch-folder/")
            return
        
        # Тестовый поиск
        print("\n3. Тестовый поиск...")
        test_query = "test"
        results = collection.query(
            query_texts=[test_query],
            n_results=min(3, count)
        )
        
        print(f"   ✓ Найдено результатов: {len(results['documents'][0])}")
        
        if results['documents'][0]:
            print("\n4. Примеры найденных документов:")
            for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
                filename = meta.get('filename', 'Unknown')
                preview = doc[:100] + "..." if len(doc) > 100 else doc
                print(f"\n   [{i}] Файл: {filename}")
                print(f"       Превью: {preview}")
        
        print("\n" + "=" * 80)
        print("✅ Тест завершен успешно!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("\nПроверьте:")
        print("  1. Docker контейнеры запущены: docker-compose ps")
        print("  2. ChromaDB доступен: curl http://localhost:8000/api/v1/heartbeat")
        print("  3. Переменные окружения в .env настроены правильно")
        sys.exit(1)


if __name__ == "__main__":
    main()
