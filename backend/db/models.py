import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure backend/packages is in sys.path
backend_packages = Path(__file__).resolve().parents[1] / "packages"
if str(backend_packages) not in sys.path:
    sys.path.insert(0, str(backend_packages))

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import relationship

from .database import Base


class User(Base):
    """
    User model for application document ownership.
    """
    __tablename__ = "users"

    id = Column(String(64), primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    documents = relationship(
        "Document",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email}>"


class Document(Base):
    """
    Document model storing uploaded document metadata, extraction status,
    and storage references linked to a user.
    """
    __tablename__ = "documents"

    id = Column(String(64), primary_key=True, index=True)
    user_id = Column(
        String(64),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename = Column(String(255), nullable=False)
    file_type = Column(String(32), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    upload_time = Column(DateTime(timezone=True), nullable=False)
    page_count = Column(Integer, nullable=True)
    extraction_status = Column(String(32), default="processing", nullable=False)
    chunk_count = Column(Integer, default=0, nullable=False)
    storage_path = Column(String(512), nullable=False)
    processed_path = Column(String(512), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    user = relationship("User", back_populates="documents")

    __table_args__ = (
        Index("ix_documents_user_created", "user_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename} status={self.extraction_status}>"
