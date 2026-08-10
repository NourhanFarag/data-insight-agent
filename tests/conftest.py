import pytest
from app.config import settings

@pytest.fixture(autouse=True)
def reset_settings():
    """Fixture to reset settings to baseline values after each test to prevent pollution."""
    # Backup
    orig_provider = settings.AI_PROVIDER
    orig_gemini_key = settings.GEMINI_API_KEY
    orig_openai_key = settings.OPENAI_API_KEY
    orig_max_rows = settings.MAX_DATASET_ROWS
    orig_max_cols = settings.MAX_DATASET_COLUMNS
    orig_max_size = settings.MAX_UPLOAD_SIZE_MB
    orig_gemini_model = settings.GEMINI_MODEL
    orig_openai_model = settings.OPENAI_MODEL

    yield

    # Restore
    settings.AI_PROVIDER = orig_provider
    settings.GEMINI_API_KEY = orig_gemini_key
    settings.OPENAI_API_KEY = orig_openai_key
    settings.MAX_DATASET_ROWS = orig_max_rows
    settings.MAX_DATASET_COLUMNS = orig_max_cols
    settings.MAX_UPLOAD_SIZE_MB = orig_max_size
    settings.GEMINI_MODEL = orig_gemini_model
    settings.OPENAI_MODEL = orig_openai_model
