# src/components/indexer_core.py
"""
Простой индексатор txt файлов в ChromaDB
"""
import os
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Set


class ChromaDBConnector:
    """Подключение к ChromaDB"""

    def __init__(self, settings):
        self.settings = settings
        self.client = None
        self.collection = None
        self.embedding_function = None

    def connect(self) -> bool:
        """Подключение к ChromaDB"""
        try:
            import chromadb
            from chromadb.config import Settings
            from chromadb.utils import embedding_functions

            url = self.settings.chroma_url
            print(f"[{self.get_timestamp()}] Подключение к ChromaDB: {url}")

            self.client = chromadb.HttpClient(
                host=self.settings.CHROMA_HOST,
                port=int(self.settings.CHROMA_PORT),
                settings=Settings(anonymized_telemetry=False)
            )

            # Embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.settings.EMBEDDINGS_MODEL
            )

            self.client.heartbeat()
            print(f"[{self.get_timestamp()}] ✓ Подключение установлено")
            return True

        except Exception as e:
            print(f"✗ Ошибка подключения: {e}")
            return False

    def get_or_create_collection(self, collection_name: str):
        """Получить или создать коллекцию"""
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            count = self.collection.count()
            print(f"[{self.get_timestamp()}] ✓ Коллекция '{collection_name}', документов: {count}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "TXT документы"}
            )
            print(f"[{self.get_timestamp()}] ✓ Создана коллекция '{collection_name}'")

        return self.collection

    @staticmethod
    def get_timestamp() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class DocumentWatcher:
    """Мониторинг и индексация txt файлов"""

    def __init__(self, settings):
        self.settings = settings
        self.watch_dir = settings.FOLDER_PATH
        self.collection_name = settings.COLLECTION_NAME
        self.connector = ChromaDBConnector(settings)
        self.collection = None
        self.state = self.load_state()

    def load_state(self) -> Dict:
        """Загрузка состояния"""
        state_file = self.settings.INDEX_STATE_FILE
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_state(self):
        """Сохранение состояния"""
        try:
            state_file = self.settings.INDEX_STATE_FILE
            # Создаём директорию если не существует
            state_dir = os.path.dirname(state_file)
            if state_dir:
                os.makedirs(state_dir, exist_ok=True)
            with open(state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"⚠ Ошибка сохранения: {e}")

    def initialize(self) -> bool:
        """Инициализация"""
        if not self.connector.connect():
            return False
        self.collection = self.connector.get_or_create_collection(self.collection_name)
        return True

    @staticmethod
    def get_timestamp() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def calculate_file_hash(self, filepath: str) -> str:
        """Хеш файла"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""

    def read_text_file(self, filepath: str) -> str:
        """Чтение txt с разными кодировками"""
        for encoding in self.settings.encodings_list:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    return f.read()
            except:
                continue
        return ""

    def scan_directory(self) -> Dict[str, Dict]:
        """Сканирование папки"""
        files_info = {}

        if not os.path.exists(self.watch_dir):
            print(f"[{self.get_timestamp()}] ⚠ Директория не существует: {self.watch_dir}")
            return files_info

        for root, dirs, files in os.walk(self.watch_dir):
            for filename in files:
                filepath = os.path.join(root, filename)
                ext = Path(filepath).suffix.lower()

                if ext not in self.settings.extensions_set:
                    continue

                try:
                    file_hash = self.calculate_file_hash(filepath)
                    files_info[filepath] = {
                        'hash': file_hash,
                        'filename': filename
                    }
                except Exception as e:
                    print(f"[{self.get_timestamp()}] ⚠ Ошибка: {filepath}: {e}")

        return files_info

    def index_file(self, filepath: str, file_info: Dict) -> bool:
        """Индексация файла"""
        try:
            filename = file_info['filename']
            print(f"[{self.get_timestamp()}] ⊙ Индексация: {filename}")

            text = self.read_text_file(filepath)

            if not text or len(text.strip()) == 0:
                print(f"[{self.get_timestamp()}]   ⚠ Пустой файл")
                return False

            # Chunking
            chunks = self._split_text(text)

            # Создаём ID и metadata
            doc_ids = []
            documents = []
            metadatas = []

            for i, chunk in enumerate(chunks):
                chunk_id = f"doc_{hashlib.md5(f'{filepath}_{i}'.encode()).hexdigest()}"
                doc_ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append({
                    'filename': filename,
                    'filepath': filepath,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'indexed_at': self.get_timestamp(),
                    'file_hash': file_info['hash']
                })

            # Удаляем старые чанки
            try:
                existing = self.collection.get(where={"filepath": filepath})
                if existing and existing['ids']:
                    self.collection.delete(ids=existing['ids'])
            except:
                pass

            # Добавляем в ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=doc_ids
            )

            self.state[filepath] = file_info['hash']

            print(f"[{self.get_timestamp()}]   ✓ Проиндексировано ({len(text)} символов, {len(chunks)} чанков)")
            return True

        except Exception as e:
            print(f"[{self.get_timestamp()}]   ✗ Ошибка: {e}")
            return False

    def _split_text(self, text: str) -> list[str]:
        """Разбиение на чанки"""
        chunk_size = self.settings.CHUNK_SIZE
        chunk_overlap = self.settings.CHUNK_OVERLAP

        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - chunk_overlap

        return chunks

    def remove_deleted_files(self, current_files: Set[str]):
        """Удаление из индекса"""
        deleted_files = set(self.state.keys()) - current_files

        if deleted_files:
            print(f"[{self.get_timestamp()}] Удалённых файлов: {len(deleted_files)}")

            for filepath in deleted_files:
                try:
                    existing = self.collection.get(where={"filepath": filepath})
                    if existing and existing['ids']:
                        self.collection.delete(ids=existing['ids'])
                    del self.state[filepath]
                    print(f"[{self.get_timestamp()}]   ✓ Удалён: {Path(filepath).name}")
                except Exception as e:
                    print(f"[{self.get_timestamp()}]   ⚠ Ошибка: {e}")

    def check_and_update(self):
        """Проверка и обновление"""
        try:
            current_files = self.scan_directory()

            if not current_files:
                return

            new_count = 0
            updated_count = 0

            for filepath, file_info in current_files.items():
                current_hash = file_info['hash']
                old_hash = self.state.get(filepath)

                if old_hash is None:
                    if self.index_file(filepath, file_info):
                        new_count += 1
                elif old_hash != current_hash:
                    if self.index_file(filepath, file_info):
                        updated_count += 1

            self.remove_deleted_files(set(current_files.keys()))

            if new_count > 0 or updated_count > 0:
                self.save_state()
                total = self.collection.count()
                print(
                    f"[{self.get_timestamp()}] ━━━ +{new_count} новых, ~{updated_count} обновлено, всего: {total} ━━━")

        except Exception as e:
            print(f"[{self.get_timestamp()}] ✗ Ошибка: {e}")

    def run_once(self):
        """Однократная проверка и обновление - для запуска через cron"""
        if not self.initialize():
            return False

        print(f"[{self.get_timestamp()}] ChromaDB:  {self.settings.chroma_url}")
        print(f"[{self.get_timestamp()}] Коллекция: {self.collection_name}")
        print(f"[{self.get_timestamp()}] Папка:     {self.watch_dir}")

        self.check_and_update()
        return True