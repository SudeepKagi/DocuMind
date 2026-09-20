import os
import sys
from pathlib import Path
from typing import Generator

# Ensure backend/packages is in sys.path
backend_packages = Path(__file__).resolve().parents[1] / "packages"
if str(backend_packages) not in sys.path:
    sys.path.insert(0, str(backend_packages))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:password@127.0.0.1:5432/documind",
)

# SQLAlchemy 2.x engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency yielding a transactional database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_or_create_dev_user(db: Session):
    """
    Retrieves or creates the default development user for document ownership in MVP.
    """
    from .models import User

    dev_user_id = os.environ.get("DEV_USER_ID", "dev_user_001")
    dev_user_email = os.environ.get("DEV_USER_EMAIL", "developer@documind.local")

    user = db.query(User).filter(User.id == dev_user_id).first()
    if not user:
        user = User(id=dev_user_id, email=dev_user_email)
        db.add(user)
        try:
            db.commit()
            db.refresh(user)
        except Exception:
            db.rollback()
            user = db.query(User).filter(User.id == dev_user_id).first()
    return user
