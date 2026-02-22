#!/usr/bin/env python3
"""
clear_collection.py — полная зачистка коллекции в ChromaDB.

Что делает:
  1. Удаляет коллекцию COLLECTION_NAME из ChromaDB
  2. Создаёт её заново (пустую)
  3. Очищает INDEX_STATE_FILE (сбрасывает хэши файлов)

После запуска следующий запуск indexer.py переиндексирует всё с нуля.

Запуск:
    python clear_collection.py                  # с подтверждением
    python clear_collection.py --force          # без подтверждения
    python clear_collection.py --state-only     # только сбросить хэши, не трогать ChromaDB
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from config.settings import settings


# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except ImportError:
#     pass


CHROMA_HOST      = settings.CHROMA_HOST
CHROMA_PORT      = int(settings.CHROMA_PORT)
COLLECTION_NAME  = settings.COLLECTION_NAME
INDEX_STATE_FILE = Path(settings.INDEX_STATE_FILE)


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def clear_chroma(force: bool = False):
    import chromadb

    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    # Проверяем существование коллекции
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME not in existing:
        log(f"Коллекция '{COLLECTION_NAME}' не найдена в ChromaDB — ничего удалять не нужно.")
    else:
        # Считаем документы перед удалением
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()

        if not force:
            answer = input(
                f"\nКоллекция '{COLLECTION_NAME}' содержит {count} документов.\n"
                f"Удалить безвозвратно? [yes/N]: "
            ).strip().lower()
            if answer != "yes":
                log("Отменено.")
                sys.exit(0)

        client.delete_collection(COLLECTION_NAME)
        log(f"Коллекция '{COLLECTION_NAME}' удалена ({count} документов).")

        # Создаём пустую коллекцию заново
        client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        log(f"Коллекция '{COLLECTION_NAME}' создана заново (пустая).")


def clear_state(force: bool = False):
    if not INDEX_STATE_FILE.exists():
        log(f"Файл состояния не найден: {INDEX_STATE_FILE} — пропускаем.")
        return

    with open(INDEX_STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)

    count = len(state)

    if not force:
        answer = input(
            f"\nФайл состояния содержит записи о {count} файлах.\n"
            f"Сбросить? [yes/N]: "
        ).strip().lower()
        if answer != "yes":
            log("Отменено.")
            sys.exit(0)

    with open(INDEX_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({}, f)

    log(f"INDEX_STATE_FILE сброшен: {INDEX_STATE_FILE} ({count} записей удалено).")


def main():
    parser = argparse.ArgumentParser(description="Зачистка коллекции ChromaDB")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Не запрашивать подтверждение",
    )
    parser.add_argument(
        "--state-only",
        action="store_true",
        help="Только сбросить INDEX_STATE_FILE, не трогать ChromaDB",
    )
    args = parser.parse_args()

    log("=" * 60)
    log(f"ChromaDB: {CHROMA_HOST}:{CHROMA_PORT} / коллекция: {COLLECTION_NAME}")
    log(f"INDEX_STATE_FILE: {INDEX_STATE_FILE}")
    log("=" * 60)

    if args.state_only:
        clear_state(force=args.force)
    else:
        clear_chroma(force=args.force)
        clear_state(force=args.force)

    log("Готово. Запустите indexer.py для переиндексации.")


if __name__ == "__main__":
    main()