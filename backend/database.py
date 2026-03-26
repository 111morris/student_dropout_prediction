"""
Database Engine & Session Management.

Uses SQLAlchemy ORM with SQLite for development.
To upgrade to PostgreSQL for production:
  1. pip install psycopg2-binary
  2. Set DATABASE_URL=postgresql://user:password@host:5432/student_ews
  3. Everything else stays the same — SQLAlchemy handles the dialect switch.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from backend.config import DATABASE_URL

# ──────────────────────────────────────────────────────────────
# Engine Configuration
# ──────────────────────────────────────────────────────────────
# SQLite needs check_same_thread=False for FastAPI's async context.
# PostgreSQL does not need this argument.
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, connect_args=connect_args, echo=False)

# workspace for the database operation 
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# ──────────────────────────────────────────────────────────────
# Dependency: get_db
# when called it will create a new session and close it after the request finishes
# ──────────────────────────────────────────────────────────────
def get_db():
    """
    FastAPI dependency that yields a database session.
    Automatically closes the session when the request finishes.

    Usage in routers:
        @router.get("/students")
        def list_students(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """
    Create all tables defined by ORM models.
    Called once on application startup.

    NOTE: For production with PostgreSQL, consider using Alembic
    for schema migrations instead of create_all():
        pip install alembic
        alembic init alembic
        alembic revision --autogenerate -m "initial"
        alembic upgrade head
    """
    Base.metadata.create_all(bind=engine)
