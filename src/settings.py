from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

current_file = Path(__file__).resolve()

current_dir = current_file.parent

env_path = current_dir.parent / ".env"


class Settings(BaseSettings):
    LANGSMITH_TRACING: str
    LANGSMITH_ENDPOINT: str
    LANGSMITH_API_KEY: str
    LANGSMITH_PROJECT: str
    OPENAI_MODEL: str
    BASE_URL: str
    EMBEDDINGS_MODEL: str
    CHROMA_HOST: str
    CHROMA_PORT: str
    COLLECTION_NAME: str
    FOLDER_PATH: str
    CHUNK_SIZE: str
    CHUNK_OVERLAP: str
    OPENAI_API_KEY: SecretStr

    model_config = SettingsConfigDict(env_file=env_path, env_file_encoding="utf-8")


settings = Settings()

if __name__ == "__main__":
    print(settings.model_dump())