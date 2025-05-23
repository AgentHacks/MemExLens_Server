# test_pinecone.py - Updated for Pinecone v7
from pinecone import (Pinecone, ServerlessSpec)
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("🧪 Testing Pinecone v7 Connection...")

try:
    # Initialize Pinecone client
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    print("✅ Connected to Pinecone!")
    
    # List existing indexes
    indexes = pc.list_indexes()
    index_names = [idx['name'] for idx in indexes]
    print(f"\n📋 Your indexes: {index_names}")
    
    # Index configuration
    index_name = "browser-history-prototype"
    
    if index_name in index_names:
        # Get index info
        index_info = pc.describe_index(index_name)
        print(f"\n📍 Index '{index_name}' Details:")
        print(f"   Host: {index_info['host']}")
        print(f"   Dimension: {index_info['dimension']}")
        print(f"   Metric: {index_info['metric']}")
        print(f"   Spec: {index_info['spec']}")
        
        # Connect to index
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print(f"   Total vectors: {stats['total_vector_count']}")
        
    else:
        print(f"\n📝 Index '{index_name}' not found.")
        print("Would you like to create it? (y/n): ", end="")
        
        if input().lower() == 'y':
            print("Creating index in Iowa (us-central1)...")
            
            pc.create_index(
                name=index_name,
                dimension=768,  # Gemini embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="gcp",
                    region="us-central1"  # Iowa
                )
            )
            print("✅ Index created successfully!")
    
    print("\n✅ All tests passed! Ready to run the Streamlit app.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print(f"Error type: {type(e).__name__}")
    
    # Detailed debugging
    if "api_key" in str(e).lower():
        print("\n🔧 API Key Issue:")
        print("1. Check your .env file has PINECONE_API_KEY")
        print("2. Make sure the key is valid")
        api_key = os.getenv("PINECONE_API_KEY")
        if api_key:
            print(f"3. Key found (starts with): {api_key[:10]}...")
        else:
            print("3. No API key found in environment!")
    
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()