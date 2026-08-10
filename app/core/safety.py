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
