from fastapi import Request
from fastapi.responses import JSONResponse

class AppBaseException(Exception):
    """Base exception for application errors."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

class DatasetValidationError(AppBaseException):
    """Raised when dataset schema, column, or row limits/validation fail."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message, status_code=status_code)

class AnalysisExecutionError(AppBaseException):
    """Raised when executing an operation fails or has invalid inputs."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message, status_code=status_code)

class FileSafetyError(AppBaseException):
    """Raised when file extension, upload size, or other safety checks fail."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message, status_code=status_code)

# Phase 2 Exceptions

class ProviderError(AppBaseException):
    """Raised when LLM calls fail, time out, or returns invalid schemas."""
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message, status_code=status_code)

class PlanValidationError(AppBaseException):
    """Raised when the LLM-generated plan contains invalid, duplicate, or unsafe actions."""
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message, status_code=status_code)

class GroundingValidationError(AppBaseException):
    """Raised when the LLM report fails structural validation/traceability verification."""
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message, status_code=status_code)

async def app_exception_handler(request: Request, exc: AppBaseException) -> JSONResponse:
    """FastAPI handler to return clean error responses without leaking implementation details."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "detail": exc.message
        }
    )
