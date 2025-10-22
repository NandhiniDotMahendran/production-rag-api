"""
FastAPI Application - Production RAG API
Day 3: Production-ready RAG system with REST API
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import os
from pathlib import Path
import shutil
from dotenv import load_dotenv
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from ProductionRag_system import ProductionRAG
from models import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    ErrorResponse
)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Production RAG API",
    description="Production-ready Retrieval Augmented Generation API with caching and streaming",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc
)

# CORS middleware (allow all origins for demo - restrict in production!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance
rag_system: ProductionRAG = None
UPLOAD_DIR = Path("week3/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system
    
    print("\n" + "="*70)
    print("🚀 Starting Production RAG API")
    print("="*70)
    
    try:
        # Initialize RAG system
        rag_system = ProductionRAG()
        
        # Check if there's a default document to load
        default_doc = "data/meta.pdf"
        if os.path.exists(default_doc):
            print(f"\n📄 Loading default document: {default_doc}")
            rag_system.load_documents(default_doc)
        else:
            print("\n⚠️  No default document found. Upload a document via /upload endpoint")
        
        print("\n✅ RAG API is ready!")
        print("📖 API Documentation: http://localhost:8000/docs")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Failed to initialize RAG system: {e}")
        raise

# Root endpoint
@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint"""
    return {
        "message": "Welcome to Production RAG API! 🚀",
        "documentation": "/docs",
        "health_check": "/health",
        "version": "1.0.0"
    }

# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Check API health status
    
    Returns system status, loaded documents, and cache statistics
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    return rag_system.health_check()

# Query endpoint (non-streaming)
@app.post(
    "/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    },
    tags=["RAG"]
)
async def query(request: QueryRequest):
    """
    Query the RAG system
    
    - **question**: Your question (1-500 characters)
    - **top_k**: Number of chunks to retrieve (1-10, default: 3)
    - **use_cache**: Whether to use cached responses (default: true)
    
    Returns the answer with sources and performance metrics
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not rag_system.chunks:
        raise HTTPException(
            status_code=400,
            detail="No documents loaded. Please upload a document first via /upload"
        )
    
    try:
        # Query the RAG system
        result = rag_system.query(
            question=request.question,
            top_k=request.top_k,
            use_cache=request.use_cache,
            stream=False
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

# Streaming query endpoint
@app.post("/query/stream", tags=["RAG"])
async def query_stream(request: QueryRequest):
    """
    Query the RAG system with streaming response
    
    Returns a stream of tokens as they are generated
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not rag_system.chunks:
        raise HTTPException(
            status_code=400,
            detail="No documents loaded. Please upload a document first via /upload"
        )
    
    try:
        # Retrieve chunks first
        chunks = rag_system.retrieve(request.question, top_k=request.top_k)
        
        if not chunks:
            async def no_results_stream():
                yield "No relevant information found to answer your question."
            return StreamingResponse(no_results_stream(), media_type="text/plain")
        
        # Build context
        context = "\n\n".join([
            f"[Source {i+1}] {chunk['chunk']}"
            for i, chunk in enumerate(chunks)
        ])
        
        # Build prompt
        prompt = f"""Based on the following context, answer the question clearly and concisely.

Context:
{context}

Question: {request.question}

Answer:"""
        
        # Create streaming response
        async def generate_stream():
            """Generator for streaming response"""
            try:
                stream = rag_system.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=rag_system.llm_model,
                    temperature=0.1,
                    max_tokens=1024,
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                        
            except Exception as e:
                yield f"\n\nError: {str(e)}"
        
        return StreamingResponse(generate_stream(), media_type="text/plain")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming query failed: {str(e)}")

# Document upload endpoint
@app.post("/upload", tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload a PDF document to the RAG system
    
    - **file**: PDF file to upload (max 10MB recommended)
    
    The document will be processed and added to the RAG system
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"📄 Uploaded file: {file.filename}")
        
        # Load document into RAG system
        num_chunks = rag_system.load_documents(str(file_path))
        
        return {
            "message": "Document uploaded and processed successfully",
            "filename": file.filename,
            "chunks_created": num_chunks,
            "status": "ready"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        )
    finally:
        await file.close()

# Clear cache endpoint
@app.post("/cache/clear", tags=["Cache"])
async def clear_cache():
    """
    Clear the query cache
    
    Useful for testing or when you want fresh responses
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    cache_size = len(rag_system.query_cache)
    rag_system.query_cache.clear()
    rag_system.cache_hits = 0
    rag_system.cache_misses = 0
    
    return {
        "message": "Cache cleared successfully",
        "cleared_entries": cache_size
    }

# Cache statistics endpoint
@app.get("/cache/stats", tags=["Cache"])
async def cache_stats():
    """
    Get cache statistics
    
    Returns cache hit rate and size information
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    return rag_system.get_cache_stats()

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )

# Run with: uvicorn main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    
    print("\n🚀 Starting FastAPI server...")
    print("📖 API Docs will be available at: http://localhost:8000/docs")
    print("🔄 Auto-reload enabled for development\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )