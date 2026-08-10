import io
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form
from app.models.responses import AnalysisResponse
from app.core.exceptions import DatasetValidationError
from app.core.safety import validate_file_metadata, validate_dataframe
from app.services.agent_service import DataInsightAgent

router = APIRouter()

@router.get("/health")
def health() -> dict:
    """Service health check endpoint."""
    return {
        "status": "ok",
        "service": "Data Insight & Decision Agent"
    }

@router.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze(
    file: UploadFile = File(...),
    question: str = Form(...)
) -> AnalysisResponse:
    """Accepts a CSV dataset upload and question, validates safety limits,

    and runs the complete agent orchestration planning/execution loop.
    """
    # 1. Validate the question
    if not question or not question.strip():
        raise DatasetValidationError("Question cannot be empty.", status_code=400)

    # 2. Get file size and validate metadata before reading (avoid OOMs)
    filename = file.filename or ""
    if not filename:
        raise DatasetValidationError("File name is missing or invalid.", status_code=400)

    file_size_bytes = file.size
    if file_size_bytes is None:
        try:
            file.file.seek(0, 2)
            file_size_bytes = file.file.tell()
            file.file.seek(0)
        except Exception:
            file_size_bytes = 0

    validate_file_metadata(filename, file_size_bytes)

    # 3. Read and decode content safely
    content_bytes = await file.read()
    try:
        content_str = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise DatasetValidationError("Failed to decode CSV. Ensure file is encoded in UTF-8.", status_code=400)

    # 4. Parse CSV safely
    try:
        df = pd.read_csv(io.StringIO(content_str))
    except Exception as e:
        raise DatasetValidationError(f"Failed to parse CSV file: {str(e)}", status_code=400)

    # 5. Validate row and column limits
    validate_dataframe(df)

    # 6. Run agent orchestration
    agent = DataInsightAgent()
    response = await agent.analyze(df, question)

    return response
