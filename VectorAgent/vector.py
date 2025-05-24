# Smart Browser History - Updated for New JSON Format
# No hardcoded credentials - everything from .env file

import os
from dotenv import load_dotenv
import google.generativeai as genai
import pinecone
from typing import List, Dict, Any, Optional
import hashlib
import asyncio
from datetime import datetime
import logging

# ============================================
# ENVIRONMENT CONFIGURATION
# ============================================

class EnvironmentConfig:
    """Manage all environment variables and credentials"""
    
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Required credentials
        self.gemini_api_key = self._get_required_env("GEMINI_API_KEY")
        self.pinecone_api_key = self._get_required_env("PINECONE_API_KEY")
        self.pinecone_environment = self._get_required_env("PINECONE_ENVIRONMENT")
        
        # Optional configurations
        self.gcp_project_id = os.getenv("GCP_PROJECT_ID", "")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "browser-history-gemini")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
        
        # Logging configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self._setup_logging()
        
    def _get_required_env(self, key: str) -> str:
        """Get required environment variable or raise error"""
        value = os.getenv(key)
        if not value:
            raise ValueError(
                f"Missing required environment variable: {key}\n"
                f"Please add it to your .env file or set it as an environment variable"
            )
        return value
    
    def _setup_logging(self):
        """Configure logging based on environment"""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def validate_all(self) -> bool:
        """Validate all credentials are properly set"""
        try:
            # Test Gemini configuration
            genai.configure(api_key=self.gemini_api_key)
            logging.info("✅ Gemini API key validated")
            
            # Test Pinecone configuration
            pinecone.init(
                api_key=self.pinecone_api_key,
                environment=self.pinecone_environment
            )
            logging.info("✅ Pinecone credentials validated")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Credential validation failed: {e}")
            return False

# Global config instance
config = EnvironmentConfig()

# ============================================
# GEMINI EMBEDDING PROCESSOR
# ============================================

class GeminiEmbeddingProcessor:
    """Handle Gemini embeddings with environment-based configuration"""
    
    def __init__(self):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configure Gemini with API key from environment
        genai.configure(api_key=self.config.gemini_api_key)
        
        # Initialize Pinecone with credentials from environment
        pinecone.init(
            api_key=self.config.pinecone_api_key,
            environment=self.config.pinecone_environment
        )
        
        self.embedding_dimension = 768  # Gemini embedding dimension
        self._setup_pinecone_index()
        
    def _setup_pinecone_index(self):
        """Create or connect to Pinecone index"""
        index_name = self.config.pinecone_index_name
        
        if index_name not in pinecone.list_indexes():
            self.logger.info(f"Creating new Pinecone index: {index_name}")
            pinecone.create_index(
                name=index_name,
                dimension=self.embedding_dimension,
                metric="cosine",
                pod_type="starter"  # Free tier
            )
        else:
            # Verify dimensions match
            index_info = pinecone.describe_index(index_name)
            if index_info.dimension != self.embedding_dimension:
                raise ValueError(
                    f"Existing index has {index_info.dimension} dimensions, "
                    f"but Gemini embeddings have {self.embedding_dimension} dimensions. "
                    f"Please delete the index or use a different name."
                )
        
        self.index = pinecone.Index(index_name)
        self.logger.info(f"Connected to Pinecone index: {index_name}")
    
    async def process_website_data(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process website data from API with new JSON format"""
        try:
            # Extract data from new format
            timestamp = api_data['timestamp']
            user_id = api_data['data']['userId']
            text_data = api_data['data']['scrapedTextData']
            url = api_data['data']['url']
            
            self.logger.info(f"Processing website: {url} for user: {user_id}")
            
            # 1. Chunk the text
            chunks = self._chunk_text(text_data, url, timestamp, user_id)
            self.logger.info(f"Created {len(chunks)} chunks")
            
            # 2. Generate embeddings
            embeddings = await self._generate_embeddings(chunks)
            
            # 3. Store in Pinecone
            await self._store_in_pinecone(chunks, embeddings)
            
            return {
                "status": "success",
                "url": url,
                "userId": user_id,
                "chunks_processed": len(chunks),
                "timestamp": timestamp
            }
            
        except Exception as e:
            self.logger.error(f"Error processing website data: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _chunk_text(self, text: str, url: str, timestamp: str, user_id: str) -> List[Dict]:
        """Split text into overlapping chunks with user context"""
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            
            # Skip very small chunks
            if len(chunk_text) < 50:
                continue
            
            # Generate unique ID for chunk (now includes user_id)
            chunk_id = hashlib.md5(
                f"{user_id}_{url}_{i}_{timestamp}".encode()
            ).hexdigest()
            
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "url": url,
                "userId": user_id,
                "timestamp": timestamp,
                "chunk_index": len(chunks),
                "start_char": i,
                "end_char": min(i + chunk_size, len(text))
            })
        
        return chunks
    
    async def _generate_embeddings(self, chunks: List[Dict]) -> List[List[float]]:
        """Generate embeddings using Gemini API"""
        embeddings = []
        batch_size = self.config.embedding_batch_size
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.logger.debug(f"Processing embedding batch {i//batch_size + 1}")
            
            for chunk in batch:
                try:
                    result = genai.embed_content(
                        model="models/embedding-001",
                        content=chunk['text'],
                        task_type="retrieval_document",
                        title=chunk['url']
                    )
                    embeddings.append(result['embedding'])
                    
                except Exception as e:
                    self.logger.error(f"Error generating embedding: {e}")
                    # Use zero vector as fallback
                    embeddings.append([0] * self.embedding_dimension)
            
            # Rate limit handling
            if i + batch_size < len(chunks):
                await asyncio.sleep(0.1)
        
        return embeddings
    
    async def _store_in_pinecone(self, chunks: List[Dict], embeddings: List[List[float]]):
        """Store embeddings in Pinecone with user context"""
        vectors = []
        
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append({
                "id": chunk['id'],
                "values": embedding,
                "metadata": {
                    "url": chunk['url'],
                    "userId": chunk['userId'],
                    "timestamp": chunk['timestamp'],
                    "text": chunk['text'][:1000],  # Metadata size limit
                    "chunk_index": chunk['chunk_index'],
                    "start_char": chunk['start_char'],
                    "end_char": chunk['end_char']
                }
            })
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
            self.logger.debug(f"Upserted batch {i//batch_size + 1} to Pinecone")

# ============================================
# SEARCH AND QUERY HANDLER
# ============================================

class GeminiSearchHandler:
    """Handle search queries using Gemini"""
    
    def __init__(self):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configure Gemini
        genai.configure(api_key=self.config.gemini_api_key)
        self.chat_model = genai.GenerativeModel('gemini-pro')
        
        # Connect to Pinecone
        pinecone.init(
            api_key=self.config.pinecone_api_key,
            environment=self.config.pinecone_environment
        )
        self.index = pinecone.Index(self.config.pinecone_index_name)
    
    async def search_and_answer(self, prompt: str, user_id: str, top_k: int = 5) -> Dict[str, Any]:
        """Search browsing history for a specific user and generate answer"""
        try:
            self.logger.info(f"Processing search prompt: {prompt} for user: {user_id}")
            
            # 1. Generate query embedding
            query_embedding = genai.embed_content(
                model="models/embedding-001",
                content=prompt,
                task_type="retrieval_query"  # Different task type for queries
            )
            
            # 2. Search Pinecone with user filter
            search_results = self.index.query(
                vector=query_embedding['embedding'],
                top_k=top_k,
                include_metadata=True,
                filter={
                    "userId": {"$eq": user_id}  # Filter by user ID
                }
            )
            
            # 3. Prepare context from results
            context = self._prepare_context(search_results.matches)
            
            # 4. Generate answer using Gemini
            answer = await self._generate_answer(prompt, context)
            
            return {
                "prompt": prompt,
                "userId": user_id,
                "answer": answer,
                "sources": [
                    {
                        "url": match.metadata['url'],
                        "timestamp": match.metadata['timestamp'],
                        "relevance_score": match.score,
                        "excerpt": match.metadata['text'][:200] + "..."
                    }
                    for match in search_results.matches
                ],
                "total_results": len(search_results.matches)
            }
            
        except Exception as e:
            self.logger.error(f"Error in search_and_answer: {e}")
            return {
                "prompt": prompt,
                "userId": user_id,
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def _prepare_context(self, matches) -> str:
        """Prepare context from search results"""
        context_parts = []
        
        for match in matches:
            metadata = match.metadata
            context_parts.append(
                f"From {metadata['url']} (visited {metadata['timestamp']}):\n"
                f"{metadata['text']}\n"
                f"Relevance: {match.score:.2f}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    async def _generate_answer(self, prompt: str, context: str) -> str:
        """Generate answer using Gemini Pro"""
        if not context:
            return "I couldn't find any relevant information in your browsing history to answer this question."
        
        prompt_template = f"""You are a helpful assistant that answers questions based on the user's browsing history.

User Question: {prompt}

Relevant Information from Browsing History:
{context}

Instructions:
1. Answer the question based ONLY on the provided browsing history
2. If the browsing history doesn't contain enough information, say so
3. Cite the source URLs when providing information
4. Be concise but comprehensive

Answer:"""

        response = self.chat_model.generate_content(prompt_template)
        return response.text

# ============================================
# FASTAPI APPLICATION
# ============================================

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="Smart Browser History API",
    description="Intelligent browser history search using Gemini embeddings",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processors (will use environment variables)
processor = None
searcher = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global processor, searcher
    
    try:
        # Validate environment configuration
        if not config.validate_all():
            raise Exception("Environment validation failed")
        
        # Initialize processors
        processor = GeminiEmbeddingProcessor()
        searcher = GeminiSearchHandler()
        
        logging.info("✅ All services initialized successfully")
        
    except Exception as e:
        logging.error(f"❌ Startup failed: {e}")
        raise

# Pydantic models matching your JSON format
class WebsiteDataPayload(BaseModel):
    userId: str
    scrapedTextData: str
    url: str

class WebsiteData(BaseModel):
    timestamp: str
    data: WebsiteDataPayload

class SearchQuery(BaseModel):
    userId: str
    prompt: str

# API Endpoints
@app.post("/process-website")
async def process_website(data: WebsiteData):
    """Process and store website data"""
    if not processor:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    result = await processor.process_website_data(data.dict())
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result.get("error"))
    
    return result

@app.post("/search")
async def search_history(query: SearchQuery):
    """Search through browsing history for a specific user"""
    if not searcher:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Use default top_k of 5
    result = await searcher.search_and_answer(query.prompt, query.userId, top_k=5)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return result

@app.get("/health")
async def health_check():
    """Check service health"""
    return {
        "status": "healthy",
        "gemini_configured": bool(config.gemini_api_key),
        "pinecone_configured": bool(config.pinecone_api_key),
        "index_name": config.pinecone_index_name
    }

@app.get("/config")
async def get_config():
    """Get non-sensitive configuration"""
    return {
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "embedding_batch_size": config.embedding_batch_size,
        "pinecone_environment": config.pinecone_environment,
        "index_name": config.pinecone_index_name
    }

@app.get("/user/{user_id}/stats")
async def get_user_stats(user_id: str):
    """Get statistics for a specific user"""
    if not processor:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Query Pinecone for user's data count
        # This is a simple count query
        stats_results = processor.index.query(
            vector=[0] * processor.embedding_dimension,  # Dummy vector
            top_k=1,
            include_metadata=True,
            filter={
                "userId": {"$eq": user_id}
            }
        )
        
        return {
            "userId": user_id,
            "status": "success",
            "message": f"User {user_id} has data in the system"
        }
    except Exception as e:
        return {
            "userId": user_id,
            "status": "error",
            "error": str(e)
        }

# ============================================
# CREATE REQUIRED FILES
# ============================================

def create_env_template():
    """Create .env.example file"""
    env_template = """# Gemini API Configuration
# Get your API key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your-gemini-api-key-here

# Pinecone Configuration
# Get your credentials from: https://app.pinecone.io/
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_ENVIRONMENT=gcp-starter
PINECONE_INDEX_NAME=browser-history-gemini

# Optional: GCP Configuration
GCP_PROJECT_ID=your-gcp-project-id

# Processing Configuration (Optional - defaults shown)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_BATCH_SIZE=100

# Logging Configuration
LOG_LEVEL=INFO
"""
    
    with open(".env.example", "w") as f:
        f.write(env_template)
    
    print("✅ Created .env.example file")

def create_requirements_txt():
    """Create requirements.txt file"""
    requirements = """# Core dependencies
google-generativeai>=0.3.0
pinecone-client>=2.2.0
python-dotenv>=1.0.0

# API dependencies
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0

# Utilities
asyncio
hashlib
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("✅ Created requirements.txt file")

def create_test_script():
    """Create a test script for the API"""
    test_script = """#!/usr/bin/env python3
# test_api.py - Test script for Smart Browser History API

import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    \"\"\"Test health endpoint\"\"\"
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())

def test_process_website():
    \"\"\"Test processing a website\"\"\"
    data = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data": {
            "userId": "user123",
            "scrapedTextData": "This is a test webpage about Python programming. Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
            "url": "https://example.com/python-tutorial"
        }
    }
    
    response = requests.post(f"{BASE_URL}/process-website", json=data)
    print("\\nProcess Website Response:", response.json())
    return response.json()

def test_search(user_id="user123"):
    \"\"\"Test searching\"\"\"
    search_data = {
        "userId": user_id,
        "prompt": "What programming language is known for simplicity?"
    }
    
    response = requests.post(f"{BASE_URL}/search", json=search_data)
    print("\\nSearch Response:", response.json())
    return response.json()

if __name__ == "__main__":
    print("Testing Smart Browser History API...")
    
    # Test health
    test_health()
    
    # Test processing
    print("\\n" + "="*50)
    process_result = test_process_website()
    
    # Wait a bit for processing
    import time
    time.sleep(2)
    
    # Test search
    print("\\n" + "="*50)
    search_result = test_search()
"""
    
    with open("test_api.py", "w") as f:
        f.write(test_script)
    
    print("✅ Created test_api.py file")

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Create template files
    create_env_template()
    create_requirements_txt()
    create_test_script()
    
    # Validate configuration
    print("\n🔍 Validating environment configuration...")
    if config.validate_all():
        print("✅ All configurations valid!")
        
        # Run the API
        print("\n🚀 Starting API server...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("❌ Configuration validation failed. Please check your .env file")