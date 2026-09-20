from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    question: str = Field(..., description="Natural language query or question about documents")


class ClassificationRequest(BaseModel):
    question: str = Field(..., description="Query identifying a document to classify")


class ExtractionRequest(BaseModel):
    question: str = Field(..., description="Query identifying a document and metadata field to extract")


class SearchRequest(BaseModel):
    question: str = Field(..., description="Search query across indexed documents")
    top_k: int = Field(5, ge=1, le=50, description="Maximum number of search results to return")


class QARequest(BaseModel):
    question: str = Field(..., description="Question to answer using grounded retrieval and LLM")


class DocumentRequest(BaseModel):
    question: str = Field(..., description="Document-centric query")
