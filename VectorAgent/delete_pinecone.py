import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables from .env file
load_dotenv()

# Get Pinecone credentials from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Create a Pinecone instance
pc = Pinecone(api_key=PINECONE_API_KEY)

# Index to delete
index_name = "browser-history-prototype"

# Check and delete
index_names = [index.name for index in pc.list_indexes()]
if index_name in index_names:
    pc.delete_index(index_name)
    print(f"✅ Index '{index_name}' has been deleted.")
else:
    print(f"⚠️ Index '{index_name}' does not exist.")
