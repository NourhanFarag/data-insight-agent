from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "Data Insight & Decision Agent"
    APP_ENV: str = "development"
    AI_PROVIDER: str = "mock"
    MAX_UPLOAD_SIZE_MB: int = 5
    MAX_DATASET_ROWS: int = 10000
    MAX_DATASET_COLUMNS: int = 50

    # Phase 2 Parameters
    MAX_ANALYSIS_STEPS: int = 8

    # OpenAI Settings
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-5.6-luna"

    # Gemini Settings
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.6-flash"

    # Ollama Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3:8b"
    OLLAMA_TEMPERATURE: float = 0.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
