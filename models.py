"""
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class QueryRequest(BaseModel):
    """Request model for /query endpoint"""
    question: str = Field(..., min_length=1, max_length=500, description="User question")
    top_k: int = Field(3, ge=1, le=10, description="Number of chunks to retrieve")
    use_cache: bool = Field(True, description="Whether to use cache")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is MetaGPT?",
                "top_k": 3,
                "use_cache": True
            }
        }

class SourceInfo(BaseModel):
    """Source information for retrieved chunks"""
    text: str = Field(..., description="Chunk text (truncated)")
    score: float = Field(..., description="Similarity score")
    chunk_id: int = Field(..., description="Chunk identifier")

class QueryResponse(BaseModel):
    """Response model for /query endpoint"""
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceInfo] = Field(..., description="Retrieved source chunks")
    retrieval_time_ms: float = Field(..., description="Retrieval time in milliseconds")
    generation_time_ms: Optional[float] = Field(None, description="Generation time in milliseconds")
    total_time_ms: float = Field(..., description="Total time in milliseconds")
    cached: bool = Field(..., description="Whether response was cached")
    cache_stats: Optional[Dict] = Field(None, description="Cache statistics")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "MetaGPT is a meta programming framework...",
                "sources": [
                    {
                        "text": "MetaGPT is a framework that...",
                        "score": 0.85,
                        "chunk_id": 0
                    }
                ],
                "retrieval_time_ms": 25.3,
                "generation_time_ms": 2341.2,
                "total_time_ms": 2366.5,
                "cached": False,
                "cache_stats": {
                    "hits": 0,
                    "misses": 1,
                    "hit_rate": "0.0%",
                    "cache_size": 0
                }
            }
        }

class HealthResponse(BaseModel):
    """Response model for /health endpoint"""
    status: str
    embedding_model: str
    llm_model: str
    chunks_loaded: int
    cache_size: int
    cache_stats: Dict

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid request",
                "detail": "Question cannot be empty"
            }
        }