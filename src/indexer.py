#!/usr/bin/env python3
# src/indexer.py
"""
Индексатор txt файлов в ChromaDB.
Запускается через cron по расписанию, выполняет одну проверку и завершается.

Пример crontab:
*/30 * * * * /path/to/.venv/bin/python /path/to/src/indexer.py >> /path/to/logs/indexer.log 2>&1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.settings import settings

# Пробуем оба варианта импорта
try:
    from src.components.indexer_core import DocumentWatcher
except ImportError:
    from src.indexer_core import DocumentWatcher


def main():
    print("=" * 60)
    print("🚀 Индексатор запущен")
    print("=" * 60)

    watcher = DocumentWatcher(settings)
    success = watcher.run_once()

    print("=" * 60)
    print(f"{'✅ Завершено' if success else '❌ Ошибка'}")
    print("=" * 60)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()