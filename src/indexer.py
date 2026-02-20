#!/usr/bin/env python3
"""
indexer.py — сканирует FOLDER_PATH, загружает/обновляет эмбеддинги в ChromaDB.
Отслеживает изменения файлов через MD5-хэши, хранит состояние в INDEX_STATE_FILE.
Запускается вручную или по cron.

Пример cron (каждые 10 минут):
    */10 * * * * /usr/bin/python3 /path/to/indexer.py >> /var/log/indexer.log 2>&1
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# ── Загружаем .env если есть ────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv не обязателен, можно передавать env через shell

# ── Настройки из переменных окружения ───────────────────────────────────────
FOLDER_PATH       = Path(os.environ["FOLDER_PATH"])
CHROMA_HOST       = os.getenv("CHROMA_HOST")
CHROMA_PORT       = int(os.getenv("CHROMA_PORT"))
COLLECTION_NAME   = os.getenv("COLLECTION_NAME")
EMBEDDINGS_MODEL  = os.getenv("EMBEDDINGS_MODEL")
INDEX_STATE_FILE  = Path(os.getenv("INDEX_STATE_FILE"))
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP"))


# ── Вспомогательные функции ──────────────────────────────────────────────────

def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def md5_file(path: Path) -> str:
    """Считает MD5-хэш файла."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state() -> dict:
    """Загружает сохранённое состояние хэшей {filepath: md5}."""
    if INDEX_STATE_FILE.exists():
        with open(INDEX_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    """Сохраняет состояние хэшей на диск."""
    INDEX_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def scan_txt_files() -> dict[str, Path]:
    """Возвращает {str(path): Path} для всех .txt файлов в FOLDER_PATH."""
    if not FOLDER_PATH.exists():
        log(f"ERROR: FOLDER_PATH не существует: {FOLDER_PATH}")
        sys.exit(1)
    return {str(p): p for p in FOLDER_PATH.rglob("*.txt")}


def doc_ids_for_file(filepath: str, n_chunks: int) -> list[str]:
    """Генерирует стабильные IDs чанков для файла: <md5_path>_chunk_0, _chunk_1 ..."""
    base = hashlib.md5(filepath.encode()).hexdigest()
    return [f"{base}_chunk_{i}" for i in range(n_chunks)]


# ── Основная логика ──────────────────────────────────────────────────────────

def get_chroma_collection():
    """Подключается к ChromaDB и возвращает коллекцию."""
    import chromadb
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    # get_or_create — безопасно при первом запуске
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def get_embeddings_model():
    """Инициализирует модель эмбеддингов (один раз)."""
    from langchain_huggingface import HuggingFaceEmbeddings
    log(f"Загружаем модель эмбеддингов: {EMBEDDINGS_MODEL}")
    return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)


def split_file(filepath: Path) -> list:
    """Загружает и разбивает один .txt файл на чанки через LangChain."""
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = TextLoader(str(filepath), encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def delete_file_chunks(collection, filepath: str):
    """Удаляет все чанки файла из коллекции по метаданным source."""
    results = collection.get(where={"source": filepath})
    ids_to_delete = results.get("ids", [])
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        log(f"  Удалено чанков: {len(ids_to_delete)} для {Path(filepath).name}")


def upsert_file(collection, embeddings_model, filepath: Path):
    """Разбивает файл на чанки и загружает в ChromaDB."""
    chunks = split_file(filepath)
    if not chunks:
        log(f"  WARN: нет чанков в {filepath.name}")
        return 0

    texts = [c.page_content for c in chunks]
    vectors = embeddings_model.embed_documents(texts)

    # Стабильные IDs на основе пути файла + порядкового номера чанка
    ids = doc_ids_for_file(str(filepath), len(chunks))

    metadatas = [
        {
            "source": str(filepath),
            "filename": filepath.name,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas,
    )
    return len(chunks)


def run():
    log("=" * 60)
    log(f"Старт индексации: {FOLDER_PATH}")
    log(f"ChromaDB: {CHROMA_HOST}:{CHROMA_PORT} / коллекция: {COLLECTION_NAME}")

    # Подключаемся к ChromaDB
    _, collection = get_chroma_collection()
    log(f"Документов в коллекции до старта: {collection.count()}")

    # Загружаем сохранённые хэши
    state = load_state()

    # Сканируем папку
    current_files = scan_txt_files()
    log(f"Найдено .txt файлов: {len(current_files)}")

    # Инициализируем модель только если есть что делать
    files_to_update = []
    for filepath_str, filepath in current_files.items():
        current_hash = md5_file(filepath)
        saved_hash = state.get(filepath_str)
        if current_hash != saved_hash:
            files_to_update.append((filepath_str, filepath, current_hash))

    # Удалённые файлы (были в state, нет в папке)
    deleted_files = [fp for fp in state if fp not in current_files]

    if not files_to_update and not deleted_files:
        log("Изменений нет. Выход.")
        return

    # Загружаем модель эмбеддингов только если нужно
    embeddings_model = get_embeddings_model()

    # Обрабатываем изменённые / новые файлы
    for filepath_str, filepath, new_hash in files_to_update:
        action = "Обновление" if filepath_str in state else "Добавление"
        log(f"{action}: {filepath.name}")

        # Удаляем старые чанки этого файла (если были)
        delete_file_chunks(collection, filepath_str)

        # Загружаем новые чанки
        n = upsert_file(collection, embeddings_model, filepath)
        log(f"  Загружено чанков: {n}")

        # Обновляем хэш в state
        state[filepath_str] = new_hash

    # Удаляем чанки удалённых файлов
    for filepath_str in deleted_files:
        log(f"Удаление (файл пропал): {Path(filepath_str).name}")
        delete_file_chunks(collection, filepath_str)
        del state[filepath_str]

    # Сохраняем обновлённый state
    save_state(state)

    log(f"Документов в коллекции после: {collection.count()}")
    log("Индексация завершена.")
    log("=" * 60)


if __name__ == "__main__":
    run()