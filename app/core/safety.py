import pandas as pd
from app.config import settings
from app.core.exceptions import FileSafetyError, DatasetValidationError

def validate_file_metadata(filename: str, content_length: int) -> None:
    """Validate file extension and raw byte size before reading the file."""
    # Check extension
    if not filename.lower().endswith(".csv"):
        raise FileSafetyError("Invalid file extension. Only CSV files are supported.", status_code=400)

    # Check if empty (0 bytes)
    if content_length <= 0:
        raise FileSafetyError("Uploaded file is empty.", status_code=400)

    # Check max size limit
    max_size_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if content_length > max_size_bytes:
        raise FileSafetyError(
            f"File size exceeds the limit of {settings.MAX_UPLOAD_SIZE_MB}MB.",
            status_code=400
        )

def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate dataset structure, emptiness, rows and columns limits after parsing."""
    if df.empty:
        raise DatasetValidationError("Dataset is empty.", status_code=400)

    rows, cols = df.shape
    if rows == 0:
        raise DatasetValidationError("Dataset has zero rows.", status_code=400)

    if rows > settings.MAX_DATASET_ROWS:
        raise DatasetValidationError(
            f"Dataset row count ({rows}) exceeds the maximum limit of {settings.MAX_DATASET_ROWS}.",
            status_code=400
        )

    if cols > settings.MAX_DATASET_COLUMNS:
        raise DatasetValidationError(
            f"Dataset column count ({cols}) exceeds the maximum limit of {settings.MAX_DATASET_COLUMNS}.",
            status_code=400
        )


import re
from typing import Any, List

def _sanitize_string(s: str) -> str:
    if not isinstance(s, str):
        return str(s)
    s_lower = s.lower()
    # Check for prompt injection or raw CSV cell leak
    suspicious_inj = ["ignore", "system", "instruction", "secret", "whoami", "os.system"]
    if ";" in s or "--" in s or "drop" in s_lower or "select" in s_lower or "union" in s_lower or "delete" in s_lower or any(k in s_lower for k in suspicious_inj):
        return "<redacted adversarial value>"
    # Also check for suspicious quotes or length
    if len(s) > 40 or any(c in s for c in ["'", '"', ";", "--", "/*", "*/"]):
        return "<redacted categorical value>"
    return s

def _sanitize_value(val: Any, tags: List[str] | None = None, registry: dict | None = None) -> Any:
    """Helper to safely serialize diagnostic values (expected/actual), redacting CSV cell payloads."""
    if registry is None:
        registry = {}
    is_adversarial = tags is not None and "adversarial" in tags

    if val is None:
        return None
    if isinstance(val, (int, float, bool)):
        return val
    if isinstance(val, str):
        if is_adversarial:
            if val not in registry:
                registry[val] = f"<redacted category {len(registry) + 1}>"
            return registry[val]
        return _sanitize_string(val)
    if isinstance(val, dict):
        sanitized_dict = {}
        for k, v in val.items():
            sanitized_key = _sanitize_value(k, tags, registry)
            sanitized_val = _sanitize_value(v, tags, registry)
            sanitized_dict[sanitized_key] = sanitized_val
        return sanitized_dict
    if isinstance(val, list):
        return [_sanitize_value(v, tags, registry) for v in val]
    return "<redacted categorical value>"
