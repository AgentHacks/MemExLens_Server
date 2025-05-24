import os
import google.generativeai as genai
import hashlib
import logging
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Logging setup
logger = logging.getLogger(__name__)

# Constants
EMBED_DIM = 768
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Gemini setup
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "browser-history-prototype")

pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if INDEX_NAME not in [index.name for index in pc.list_indexes()]:
    logger.info(f"Creating Pinecone index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="gcp", region=PINECONE_ENV)
    )

pinecone_index = pc.Index(INDEX_NAME)

# ================================
def embed_and_store_in_pinecone(data_obj, timestamp):
    user_id = data_obj["userId"]
    url = data_obj["url"]
    text = data_obj["scrapedTextData"]

    # Chunk text
    chunks = []
    for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk_text = text[i:i + CHUNK_SIZE]
        if len(chunk_text) < 50:
            continue
        chunk_id = hashlib.md5(f"{user_id}_{url}_{i}_{timestamp}".encode()).hexdigest()
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "url": url,
            "userId": user_id,
            "timestamp": timestamp,
            "chunk_index": len(chunks),
            "start_char": i,
            "end_char": min(i + CHUNK_SIZE, len(text))
        })

    # Embed and prepare vectors
    vectors = []
    for chunk in chunks:
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=chunk["text"],
                task_type="retrieval_document",
                title=chunk["url"]
            )
            embedding = result["embedding"]
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            embedding = [0.0] * EMBED_DIM

        vectors.append({
            "id": chunk["id"],
            "values": embedding,
            "metadata": {
                "userId": chunk["userId"],
                "url": chunk["url"],
                "timestamp": chunk["timestamp"],
                "text": chunk["text"][:1000],
                "chunk_index": chunk["chunk_index"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"]
            }
        })

    # Upsert vectors to Pinecone
    for i in range(0, len(vectors), 100):
        pinecone_index.upsert(vectors=vectors[i:i+100])

    logger.info(f"✅ Embedded and stored {len(vectors)} chunks for user {user_id}")
