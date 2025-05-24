import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables from .env
load_dotenv()

# Create Pinecone client instance
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define your index name
index_name = "browser-history-prototype"

# Check if index exists
if index_name not in pc.list_indexes().names():
    raise ValueError(f"Index '{index_name}' does not exist.")

# Connect to the index
index = pc.Index(index_name)

# Delete all vector embeddings (not the index itself)
index.delete(delete_all=True)

print(f"✅ All vectors deleted from index: '{index_name}', index still exists.")
