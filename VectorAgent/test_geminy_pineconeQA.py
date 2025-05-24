# streamlit_app.py - Smart Browser History Q&A Prototype
# Run with: streamlit run streamlit_app.py

import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import pinecone
import requests
from bs4 import BeautifulSoup
import hashlib
from datetime import datetime
import time
from typing import List, Dict, Any
import re

# Load environment variables
load_dotenv()

# Updated Pinecone import
from pinecone import Pinecone, ServerlessSpec

# ============================================
# CONFIGURATION
# ============================================

# Page config
st.set_page_config(
    page_title="Smart Browser History Q&A",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'processed_urls' not in st.session_state:
    st.session_state.processed_urls = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ============================================
# SETUP CREDENTIALS
# ============================================

@st.cache_resource
def initialize_services():
    """Initialize Gemini and Pinecone services"""
    try:
        # Get credentials
        gemini_key = os.getenv("GEMINI_API_KEY")
        pinecone_key = os.getenv("PINECONE_API_KEY")
        
        if not gemini_key or not pinecone_key:
            st.error("⚠️ Missing API keys! Please check your .env file.")
            st.stop()
        
        # Configure Gemini
        genai.configure(api_key=gemini_key)
        
        # Initialize new Pinecone client
        pc = Pinecone(api_key=pinecone_key)
        
        # Create or connect to index
        index_name = "browser-history-prototype"
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=768,  # Gemini embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="gcp",
                    region="us-central1"  # Iowa region
                )
            )
            # Wait for index to be ready
            import time
            time.sleep(2)
        
        index = pc.Index(index_name)
        
        return index, True
        
    except Exception as e:
        st.error(f"Failed to initialize services: {str(e)}")
        return None, False

# ============================================
# URL PROCESSING
# ============================================

class URLProcessor:
    """Process URLs and extract content"""
    
    @staticmethod
    def fetch_url_content(url: str) -> Dict[str, Any]:
        """Fetch and extract content from URL"""
        try:
            # Add headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.text.strip() if title else "No title"
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Clean text
            text = re.sub(r'\s+', ' ', text)
            text = text[:10000]  # Limit to 10k characters
            
            return {
                'url': url,
                'title': title_text,
                'content': text,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
        except Exception as e:
            return {
                'url': url,
                'title': 'Error',
                'content': f"Failed to fetch content: {str(e)}",
                'timestamp': datetime.now().isoformat(),
                'success': False
            }
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 100:  # Skip very small chunks
                chunks.append(chunk)
        return chunks

# ============================================
# VECTOR OPERATIONS
# ============================================

class VectorManager:
    """Manage vector embeddings and storage"""
    
    def __init__(self, index):
        self.index = index
    
    def generate_embeddings(self, texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
        """Generate embeddings using Gemini"""
        embeddings = []
        
        for text in texts:
            try:
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type=task_type
                )
                embeddings.append(result['embedding'])
            except Exception as e:
                st.error(f"Embedding error: {e}")
                embeddings.append([0] * 768)  # Fallback
        
        return embeddings
    
    def store_embeddings(self, url: str, title: str, chunks: List[str], embeddings: List[List[float]]):
        """Store embeddings in Pinecone"""
        vectors = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = hashlib.md5(f"{url}_{i}_{datetime.now()}".encode()).hexdigest()
            
            vectors.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": {
                    "url": url,
                    "title": title,
                    "chunk_text": chunk[:1000],
                    "chunk_index": i,
                    "timestamp": datetime.now().isoformat()
                }
            })
        
        # Upsert to Pinecone
        self.index.upsert(vectors=vectors)
        return len(vectors)
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar content"""
        # Generate query embedding
        query_embedding = self.generate_embeddings([query], task_type="retrieval_query")[0]
        
        # Search Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return results['matches']

# ============================================
# Q&A SYSTEM
# ============================================

class QASystem:
    """Handle question answering"""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
    
    def generate_answer(self, query: str, search_results: List[Dict]) -> str:
        """Generate answer based on search results"""
        
        if not search_results:
            return "I couldn't find any relevant information in your browsing history for this question."
        
        # Prepare context
        context_parts = []
        for match in search_results:
            metadata = match['metadata']
            context_parts.append(
                f"From: {metadata['title']} ({metadata['url']})\n"
                f"Content: {metadata['chunk_text']}\n"
                f"Relevance: {match['score']:.2f}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate answer
        prompt = f"""Based on the user's browsing history, answer their question.

Question: {query}

Relevant content from browsing history:
{context}

Instructions:
1. Answer based ONLY on the provided content
2. Be specific and cite the sources
3. If information is incomplete, mention it
4. Be concise but comprehensive

Answer:"""

        response = self.model.generate_content(prompt)
        return response.text

# ============================================
# STREAMLIT UI
# ============================================

def main():
    st.title("🔍 Smart Browser History Q&A")
    st.markdown("Process URLs and ask questions about the content!")
    
    # Initialize services
    index, initialized = initialize_services()
    if not initialized:
        st.stop()
    
    # Create managers
    vector_manager = VectorManager(index)
    qa_system = QASystem()
    url_processor = URLProcessor()
    
    # Sidebar for URL processing
    with st.sidebar:
        st.header("📥 Add URLs to History")
        
        # URL input
        url_input = st.text_input("Enter URL to process:")
        
        if st.button("🚀 Process URL", type="primary"):
            if url_input:
                with st.spinner("Processing URL..."):
                    # Fetch content
                    content_data = url_processor.fetch_url_content(url_input)
                    
                    if content_data['success']:
                        # Chunk text
                        chunks = url_processor.chunk_text(content_data['content'])
                        st.info(f"Created {len(chunks)} chunks")
                        
                        # Generate embeddings
                        embeddings = vector_manager.generate_embeddings(chunks)
                        
                        # Store in Pinecone
                        count = vector_manager.store_embeddings(
                            url=content_data['url'],
                            title=content_data['title'],
                            chunks=chunks,
                            embeddings=embeddings
                        )
                        
                        # Update session state
                        st.session_state.processed_urls.append({
                            'url': content_data['url'],
                            'title': content_data['title'],
                            'timestamp': content_data['timestamp'],
                            'chunks': count
                        })
                        
                        st.success(f"✅ Processed: {content_data['title']}")
                    else:
                        st.error(f"❌ Failed: {content_data['content']}")
        
        # Show processed URLs
        st.divider()
        st.subheader("📚 Processed URLs")
        
        if st.session_state.processed_urls:
            for i, item in enumerate(st.session_state.processed_urls):
                with st.expander(f"{i+1}. {item['title'][:50]}..."):
                    st.write(f"**URL:** {item['url']}")
                    st.write(f"**Processed:** {item['timestamp'][:16]}")
                    st.write(f"**Chunks:** {item['chunks']}")
        else:
            st.info("No URLs processed yet")
        
        # Clear history button
        if st.button("🗑️ Clear All History"):
            # Clear Pinecone index
            index.delete(delete_all=True)
            st.session_state.processed_urls = []
            st.session_state.chat_history = []
            st.rerun()
    
    # Main area for Q&A
    st.header("💬 Ask Questions")
    
    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for chat in st.session_state.chat_history:
            with st.chat_message(chat["role"]):
                st.write(chat["content"])
                if "sources" in chat:
                    with st.expander("📚 Sources"):
                        for source in chat["sources"]:
                            st.write(f"- [{source['title']}]({source['url']}) (Score: {source['score']:.2f})")
    
    # Query input
    query = st.chat_input("Ask a question about your browsing history...")
    
    if query:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.write(query)
        
        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching and thinking..."):
                # Search for relevant content
                search_results = vector_manager.search_similar(query, top_k=5)
                
                # Generate answer
                answer = qa_system.generate_answer(query, search_results)
                
                # Display answer
                st.write(answer)
                
                # Show sources
                if search_results:
                    sources = [
                        {
                            'title': match['metadata']['title'],
                            'url': match['metadata']['url'],
                            'score': match['score']
                        }
                        for match in search_results
                    ]
                    
                    with st.expander("📚 Sources"):
                        for source in sources:
                            st.write(f"- [{source['title']}]({source['url']}) (Score: {source['score']:.2f})")
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })
    
    # Footer with stats
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("URLs Processed", len(st.session_state.processed_urls))
    
    with col2:
        total_chunks = sum(item['chunks'] for item in st.session_state.processed_urls)
        st.metric("Total Chunks", total_chunks)
    
    with col3:
        st.metric("Questions Asked", len([c for c in st.session_state.chat_history if c['role'] == 'user']))

# ============================================
# EXAMPLE USAGE
# ============================================

def show_example_usage():
    st.sidebar.divider()
    with st.sidebar.expander("📖 Example Usage"):
        st.markdown("""
        **Try these steps:**
        
        1. **Add a URL:**
           - https://en.wikipedia.org/wiki/Artificial_intelligence
           - https://docs.python.org/3/tutorial/
        
        2. **Ask questions:**
           - "What is artificial intelligence?"
           - "How do I use loops in Python?"
           - "What did I read about machine learning?"
        
        **Tips:**
        - Process multiple URLs to build your knowledge base
        - Ask specific questions for better results
        - Check sources to verify information
        """)

# Run the app
if __name__ == "__main__":
    main()
    show_example_usage()