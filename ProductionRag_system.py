"""
Production RAG System - Fixed PDF Extraction
Handles PDF encoding issues properly
"""

import os
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import time
from pathlib import Path
import pickle
import hashlib

class ProductionRAG:
    """Production-ready RAG system with proper PDF handling"""
    
    def __init__(
        self,
        embedding_model: str = "paraphrase-MiniLM-L3-v2",
        llm_model: str = "llama-3.3-70b-versatile",
        groq_api_key: Optional[str] = None,
        cache_dir: str = "cache"
    ):
        print(f"🚀 Initializing Production RAG...")
        
        # Create cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load embedding model
        print(f"📥 Loading embedding model: {embedding_model}")
        self.embed_model = SentenceTransformer(embedding_model)
        self.embedding_model_name = embedding_model
        
        # Initialize Groq client
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found!")
        
        self.groq_client = Groq(api_key=self.groq_api_key)
        self.llm_model = llm_model
        
        # Storage
        self.chunks: List[str] = []
        self.chunk_embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict] = []
        self.current_doc_hash: Optional[str] = None
        
        # Query cache
        self.query_cache: Dict[str, Dict] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        print("✅ RAG System initialized\n")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash of file for cache identification"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _get_cache_path(self, doc_hash: str) -> Path:
        """Get path for cached embeddings"""
        return self.cache_dir / f"embeddings_{doc_hash}.pkl"
    
    def _save_embeddings_to_cache(self, doc_hash: str):
        """Save embeddings to disk"""
        cache_data = {
            'chunks': self.chunks,
            'embeddings': self.chunk_embeddings,
            'metadata': self.metadata,
            'model': self.embedding_model_name,
            'doc_hash': doc_hash
        }
        
        cache_path = self._get_cache_path(doc_hash)
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"💾 Embeddings cached to: {cache_path.name}")
    
    def _load_embeddings_from_cache(self, doc_hash: str) -> bool:
        """Load embeddings from cache"""
        cache_path = self._get_cache_path(doc_hash)
        
        if not cache_path.exists():
            return False
        
        print(f"📂 Loading from cache...")
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            if cache_data['model'] != self.embedding_model_name:
                print(f"⚠️  Model mismatch, regenerating...")
                return False
            
            self.chunks = cache_data['chunks']
            self.chunk_embeddings = cache_data['embeddings']
            self.metadata = cache_data['metadata']
            self.current_doc_hash = cache_data['doc_hash']
            
            print(f"✅ Loaded {len(self.chunks)} chunks from cache (instant!) 🚀")
            return True
            
        except Exception as e:
            print(f"⚠️  Cache load failed: {e}")
            return False
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF with multiple fallback methods
        Handles encoding issues properly
        """
        full_text = ""
        
        # Method 1: Try PyMuPDF with proper text extraction
        try:
            import fitz
            print("   Trying PyMuPDF extraction...")
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Use get_text with 'text' layout option for better extraction
                text = page.get_text("text", sort=True)
                
                # Clean up the text
                if text and len(text.strip()) > 0:
                    full_text += text + "\n"
            
            doc.close()
            
            # Validate extraction
            if len(full_text.strip()) > 100:
                print(f"   ✅ PyMuPDF: Extracted {len(full_text)} chars from {len(doc)} pages")
                return full_text
            else:
                print(f"   ⚠️  PyMuPDF extraction too short, trying fallback...")
                
        except Exception as e:
            print(f"   ⚠️  PyMuPDF failed: {e}")
        
        # Method 2: Try pypdf
        try:
            import pypdf
            print("   Trying pypdf extraction...")
            
            reader = pypdf.PdfReader(file_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            
            if len(full_text.strip()) > 100:
                print(f"   ✅ pypdf: Extracted {len(full_text)} chars")
                return full_text
            else:
                print(f"   ⚠️  pypdf extraction too short, trying fallback...")
                
        except Exception as e:
            print(f"   ⚠️  pypdf failed: {e}")
        
        # Method 3: Try llama_index (most reliable fallback)
        try:
            from llama_index.core import SimpleDirectoryReader
            print("   Trying llama_index extraction...")
            
            documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
            full_text = "\n\n".join([doc.text for doc in documents])
            
            if len(full_text.strip()) > 100:
                print(f"   ✅ llama_index: Extracted {len(full_text)} chars")
                return full_text
                
        except Exception as e:
            print(f"   ⚠️  llama_index failed: {e}")
        
        # If all methods fail
        raise ValueError(f"Failed to extract text from PDF using all methods. File may be corrupted or image-based.")
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove null characters and PDF artifacts
        text = text.replace('\x00', '')
        text = text.replace('\u0000', '')
        
        # Remove excessive whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        return text
    
    def load_documents(self, file_path: str) -> int:
        """Load documents with proper PDF handling"""
        print(f"📄 Loading document: {file_path}")
        
        # Calculate file hash
        doc_hash = self._get_file_hash(file_path)
        
        # Try cache first
        if self._load_embeddings_from_cache(doc_hash):
            return len(self.chunks)
        
        # Cache miss - process document
        print("🔄 Processing document (first time - will be cached)...")
        
        # Extract text with fallbacks
        full_text = self._extract_text_from_pdf(file_path)
        
        # Clean the text
        full_text = self._clean_text(full_text)
        
        print(f"✅ Cleaned text: {len(full_text)} characters")
        
        # Chunk the text
        self.chunks = self._chunk_text(full_text)
        self.metadata = [
            {
                "chunk_id": i,
                "length": len(chunk),
                "source": file_path
            }
            for i, chunk in enumerate(self.chunks)
        ]
        
        print(f"✅ Created {len(self.chunks)} chunks")
        
        # Show sample chunk for verification
        if self.chunks:
            print(f"\n📝 Sample chunk (first 200 chars):")
            print(f"   {self.chunks[0][:200]}...\n")
        
        # Generate embeddings
        self._generate_embeddings()
        
        # Save to cache
        self.current_doc_hash = doc_hash
        self._save_embeddings_to_cache(doc_hash)
        
        return len(self.chunks)
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Simple chunking with overlap"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Only add if substantial
            if len(chunk.strip()) > 50:
                chunks.append(chunk.strip())
            
            start += (chunk_size - overlap)
        
        return chunks
    
    def _generate_embeddings(self):
        """Generate embeddings for all chunks"""
        print("🔄 Generating embeddings...")
        start_time = time.time()
        
        self.chunk_embeddings = self.embed_model.encode(
            self.chunks,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Generated embeddings in {elapsed:.2f}s")
        print(f"   Average: {(elapsed/len(self.chunks))*1000:.2f}ms per chunk\n")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.3
    ) -> List[Dict]:
        """Retrieve most relevant chunks"""
        if not self.chunks or self.chunk_embeddings is None:
            return []
        
        # Generate query embedding
        query_embedding = self.embed_model.encode(query, convert_to_numpy=True)
        
        # Calculate similarities
        similarities = np.dot(self.chunk_embeddings, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            
            if score >= similarity_threshold:
                results.append({
                    "chunk": self.chunks[idx],
                    "score": score,
                    "metadata": self.metadata[idx]
                })
        
        return results
    
    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict],
        stream: bool = False
    ):
        """Generate answer using LLM"""
        if not context_chunks:
            return "I couldn't find relevant information to answer your question."
        
        # Build context
        context = "\n\n".join([
            f"[Source {i+1}] {chunk['chunk']}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Build prompt
        prompt = f"""Based on the following context, answer the question clearly and concisely.

Context:
{context}

Question: {query}

Answer (be specific and cite sources when relevant):"""
        
        # Generate
        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.llm_model,
                temperature=0.1,
                max_tokens=1024,
                stream=stream
            )
            
            if stream:
                return response
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def query(
        self,
        question: str,
        top_k: int = 3,
        use_cache: bool = True,
        stream: bool = False
    ) -> Dict:
        """Complete RAG query"""
        start_time = time.time()
        
        # Check cache
        cache_key = f"{question}:{top_k}"
        if use_cache and cache_key in self.query_cache:
            self.cache_hits += 1
            cached = self.query_cache[cache_key].copy()
            cached['cached'] = True
            cached['cache_stats'] = self.get_cache_stats()
            return cached
        
        self.cache_misses += 1
        
        # Retrieve
        retrieval_start = time.time()
        chunks = self.retrieve(question, top_k=top_k)
        retrieval_time = time.time() - retrieval_start
        
        if not chunks:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "retrieval_time_ms": retrieval_time * 1000,
                "total_time_ms": (time.time() - start_time) * 1000,
                "cached": False
            }
        
        # Generate answer
        generation_start = time.time()
        
        if stream:
            return {
                "answer_stream": self.generate_answer(question, chunks, stream=True),
                "sources": chunks,
                "retrieval_time_ms": retrieval_time * 1000,
                "cached": False
            }
        else:
            answer = self.generate_answer(question, chunks, stream=False)
            generation_time = time.time() - generation_start
            
            total_time = time.time() - start_time
            
            result = {
                "answer": answer,
                "sources": [
                    {
                        "text": chunk["chunk"][:200] + "..." if len(chunk["chunk"]) > 200 else chunk["chunk"],
                        "score": chunk["score"],
                        "chunk_id": chunk["metadata"]["chunk_id"]
                    }
                    for chunk in chunks
                ],
                "retrieval_time_ms": retrieval_time * 1000,
                "generation_time_ms": generation_time * 1000,
                "total_time_ms": total_time * 1000,
                "cached": False,
                "cache_stats": self.get_cache_stats()
            }
            
            # Cache the result
            if use_cache:
                self.query_cache[cache_key] = result.copy()
            
            return result
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "cache_size": len(self.query_cache)
        }
    
    def health_check(self) -> Dict:
        """System health check"""
        return {
            "status": "healthy",
            "embedding_model": self.embedding_model_name,
            "llm_model": self.llm_model,
            "chunks_loaded": len(self.chunks),
            "cache_size": len(self.query_cache),
            "cache_stats": self.get_cache_stats(),
            "embeddings_cached": self.current_doc_hash is not None
        }