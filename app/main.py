from fastapi import FastAPI
from app.config import settings
from app.core.exceptions import AppBaseException, app_exception_handler
from app.api.routes import router

app = FastAPI(
    title=settings.APP_NAME,
    description="Deterministic analysis and decision support engine (Phase 1)",
    version="1.0.0"
)

# Register global application exception handlers
app.add_exception_handler(AppBaseException, app_exception_handler)

# Include API endpoints (mounted at root)
app.include_router(router)
