# ============================================================
# DocuMind FastAPI Backend
# ============================================================

import os
import sys
from pathlib import Path

# Prevent CUDA memory fragmentation on 6GB laptop GPUs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Add backend/packages to sys.path for pure-Python parsing packages
BACKEND_DIR = Path(__file__).resolve().parent
PACKAGES_DIR = BACKEND_DIR / "packages"
if str(PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGES_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")
except ImportError:
    pass

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db import SessionLocal, get_or_create_dev_user
from routes.agent import router as agent_router
from routes.documents import router as documents_router

logger = logging.getLogger("documind.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure default development user exists in PostgreSQL on startup
    try:
        db = SessionLocal()
        user = get_or_create_dev_user(db)
        if user:
            logger.info("Database connected. Active dev user: %s (%s)", user.id, user.email)
        db.close()
    except Exception as e:
        logger.warning("Database startup check warning: %s", e)
    yield


app = FastAPI(
    title="DocuMind API",
    description="Enterprise Document Intelligence API powered by DistilBERT, BGE, BM25, LayoutLM, and Gemini",
    version="1.0.0",
    lifespan=lifespan,
)

# ============================================================
# CORS Configuration
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Core Health & Status Endpoints
# ============================================================

@app.get("/")
def root():
    return {
        "project": "DocuMind",
        "version": "1.0.0",
        "status": "running",
        "docs_url": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "DocuMind API",
    }


# ============================================================
# Include Routers
# ============================================================

app.include_router(agent_router)
app.include_router(documents_router)