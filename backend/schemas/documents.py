from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename of the uploaded file")
    file_type: str = Field(..., description="File extension / MIME type (PDF, DOCX, TXT, EML)")
    file_size: int = Field(..., description="Size of file in bytes")
    upload_time: str = Field(..., description="ISO timestamp of upload")
    page_count: Optional[int] = Field(None, description="Number of pages if applicable")
    extraction_status: str = Field(..., description="Status of text extraction: processed, failed, partial")
    chunk_count: int = Field(0, description="Total number of chunks generated")


class DocumentListResponse(BaseModel):
    status: str = "success"
    documents: List[DocumentSummary] = []
    total: int = 0


class DocumentChunkInfo(BaseModel):
    chunk_id: str
    page: Optional[int] = None
    text: str
    word_count: Optional[int] = None


class DocumentDetailResponse(BaseModel):
    status: str = "success"
    document: DocumentSummary
    chunks: List[DocumentChunkInfo] = []


class DocumentQARequest(BaseModel):
    question: str = Field(..., description="Question to answer from the uploaded document")


class SourceChunk(BaseModel):
    chunk_id: str
    page: Optional[int] = None
    text: str
    score: Optional[float] = None


class DocumentQAResponse(BaseModel):
    status: str = "success"
    document_id: str
    question: str
    answer: str
    sources: List[SourceChunk] = []
