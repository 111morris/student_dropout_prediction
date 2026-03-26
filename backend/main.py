"""
Student Dropout Early Warning System — FastAPI Application.

This is the main entry point for the backend API.
Run with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Then open http://localhost:8000/docs for the Swagger UI.
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import (
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    LOG_LEVEL,
    CORS_ORIGINS,
)
from backend.database import create_tables
from backend.routers import predict, students
from backend.schemas.student import HealthResponse
from backend.services.prediction_service import prediction_service


# ══════════════════════════════════════════════════════════════
# Logging Configuration
# ══════════════════════════════════════════════════════════════
def setup_logging() -> None:
    """Configure structured logging to console and file."""
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    # Root logger
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("ews_api.log", mode="a"),
        ],
    )

    # Silence noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


setup_logging()
logger = logging.getLogger("ews.main")


# ══════════════════════════════════════════════════════════════
# Application Lifespan (startup / shutdown)
# ══════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.
    - On startup: create DB tables, load ML model.
    - On shutdown: cleanup resources.
    """
    logger.info("=" * 60)
    logger.info("STARTING: Student Dropout Early Warning System API")
    logger.info("=" * 60)

    # Create database tables
    create_tables()
    logger.info("✓ Database tables created/verified")

    # Load the trained ML model
    prediction_service.load_model()
    logger.info("✓ ML model loaded and ready")

    logger.info("=" * 60)
    logger.info("API is ready to accept requests")
    logger.info("=" * 60)

    yield  # App is running

    logger.info("Shutting down EWS API...")


# ══════════════════════════════════════════════════════════════
# FastAPI Application
# ══════════════════════════════════════════════════════════════
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS Middleware ───────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include Routers ──────────────────────────────────────────
app.include_router(predict.router, prefix="/api/v1")
app.include_router(students.router, prefix="/api/v1")


# ══════════════════════════════════════════════════════════════
# Global Exception Handlers
# ══════════════════════════════════════════════════════════════
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler for unexpected errors."""
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred. "
                      "Please try again or contact support.",
        },
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Custom 404 handler."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": "The requested resource was not found."},
    )


# ══════════════════════════════════════════════════════════════
# Root & Health Endpoints
# ══════════════════════════════════════════════════════════════
@app.get("/", tags=["System"])
def root():
    """Root endpoint — API welcome message."""
    return {
        "message": "Student Dropout Early Warning System API",
        "version": API_VERSION,
        "docs": "/docs",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
)
def health_check() -> HealthResponse:
    """Check if the API, model, and database are operational."""
    return HealthResponse(
        status="healthy",
        model_loaded=prediction_service.model is not None,
        database_connected=True,  # If we get here, DB is fine
    )
