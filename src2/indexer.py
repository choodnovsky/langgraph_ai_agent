#!/usr/bin/env python3
# src/indexer.py
"""
Индексатор txt файлов в ChromaDB
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src2.settings import settings
from src2.indexer_core import DocumentWatcher


def main():
    print("=" * 80)
    print("🚀 Индексатор txt → ChromaDB")
    print("=" * 80)

    # Создаем директории
    Path(settings.FOLDER_PATH).mkdir(parents=True, exist_ok=True)
    Path(settings.INDEX_STATE_FILE).parent.mkdir(parents=True, exist_ok=True)

    # Запуск
    watcher = DocumentWatcher(settings)

    try:
        watcher.start_watching()
    except KeyboardInterrupt:
        print("\n👋 Завершение...")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()