# src/settings.py

from pathlib import Path
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

current_file = Path(__file__).resolve()
current_dir = current_file.parent
env_path = current_dir.parent / ".env"


class Settings(BaseSettings):
    # OpenAI настройки
    OPENAI_MODEL: str
    BASE_URL: str
    OPENAI_API_KEY: SecretStr

    # ChromaDB настройки
    CHROMA_HOST: str = "chromadb"
    CHROMA_PORT: str = "8000"
    CHROMA_PROTOCOL: str = "http"
    COLLECTION_NAME: str = "documents"

    # Embeddings модель
    EMBEDDINGS_MODEL: str

    # Настройки индексатора
    FOLDER_PATH: str = "./wiki"
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int
    CHECK_INTERVAL: int = 60

    # Файлы состояния
    INDEX_STATE_FILE: str

    # Только txt файлы!
    SUPPORTED_EXTENSIONS: str = ".txt"
    TEXT_ENCODINGS: str = "utf-8,cp1251,latin-1"

    model_config = SettingsConfigDict(
        env_file=env_path,
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def chroma_url(self) -> str:
        return f"{self.CHROMA_PROTOCOL}://{self.CHROMA_HOST}:{self.CHROMA_PORT}"

    @property
    def extensions_set(self) -> set[str]:
        return {ext.strip() for ext in self.SUPPORTED_EXTENSIONS.split(",")}

    @property
    def encodings_list(self) -> list[str]:
        return [enc.strip() for enc in self.TEXT_ENCODINGS.split(",")]


settings = Settings()