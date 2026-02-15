# src/settings.py

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

current_file = Path(__file__).resolve()

current_dir = current_file.parent

env_path = current_dir.parent / ".env"


class Settings(BaseSettings):
    # LangSmith настройки
    LANGSMITH_TRACING: str
    LANGSMITH_ENDPOINT: str
    LANGSMITH_API_KEY: str
    LANGSMITH_PROJECT: str

    # OpenAI настройки
    OPENAI_MODEL: str
    BASE_URL: str
    EMBEDDINGS_MODEL: str
    OPENAI_API_KEY: SecretStr

    # ChromaDB настройки (используются в индексаторе)
    CHROMA_HOST: str
    CHROMA_PORT: str
    COLLECTION_NAME: str

    # Настройки индексатора
    FOLDER_PATH: str  # Папка для мониторинга документов
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int

    # Дополнительные настройки индексатора
    CHROMA_PROTOCOL: str = "http"  # http или https
    CHECK_INTERVAL: int = 60  # Интервал проверки в секундах
    INDEX_STATE_FILE: str = "./index_state.json"  # Файл состояния
    LOG_FILE: str = "./indexer.log"  # Файл логов
    MAX_FILE_SIZE: int | None = None  # Максимальный размер файла (None = без ограничений)

    # Поддерживаемые расширения файлов (через .env как строка через запятую)
    SUPPORTED_EXTENSIONS: str = ".txt,.md"  # Расширения файлов для индексации
    TEXT_ENCODINGS: str = "utf-8,cp1251,latin-1,cp866"  # Кодировки для чтения файлов

    model_config = SettingsConfigDict(env_file=env_path, env_file_encoding="utf-8")

    @property
    def extensions_set(self) -> set[str]:
        """Возвращает набор поддерживаемых расширений"""
        return {ext.strip() for ext in self.SUPPORTED_EXTENSIONS.split(",")}

    @property
    def encodings_list(self) -> list[str]:
        """Возвращает список кодировок для чтения файлов"""
        return [enc.strip() for enc in self.TEXT_ENCODINGS.split(",")]


settings = Settings()

if __name__ == "__main__":
    print(settings.model_dump())