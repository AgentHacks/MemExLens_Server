import streamlit as st
import os
import json
import textwrap
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

# ================================
# Configuration
# ================================
st.set_page_config(page_title="Smart Browser History", layout="wide")
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INDEX_NAME = "browser-history-prototype"
EMBED_DIM = 768

# ================================
# Initialize Services
# ================================
@st.cache_resource
def init_services():
    genai.configure(api_key=GEMINI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="gcp", region="us-central1")
        )
        import time; time.sleep(2)
    return pc.Index(INDEX_NAME), genai.GenerativeModel("gemini-2.5-flash-preview-04-17")

index, model = init_services()

# ================================
# Helper Functions
# ================================
def embed_text(text, task_type="retrieval_document"):
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type=task_type
    )
    return result["embedding"]

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 100:
            chunks.append(chunk)
    return chunks

def process_json_payload(json_payload):
    try:
        timestamp = json_payload["timestamp"]
        user_id = json_payload["data"]["userId"]
        url = json_payload["data"]["url"]
        text = json_payload["data"]["scrapedTextData"]

        chunks = chunk_text(text)
        embeddings = [embed_text(chunk) for chunk in chunks]

        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id": hashlib.md5(f"{user_id}_{url}_{i}_{timestamp}".encode()).hexdigest(),
                "values": embedding,
                "metadata": {
                    "userId": user_id,
                    "url": url,
                    "timestamp": timestamp,
                    "chunk_index": i,
                    "text": chunk[:1000]
                }
            })

        index.upsert(vectors=vectors)
        return f"✅ Stored {len(vectors)} chunks for user {user_id}"
    except Exception as e:
        return f"❌ Error processing payload: {str(e)}"

def search_and_respond(query: str, user_id: str, top_k=5):
    query_vector = embed_text(query, task_type="retrieval_query")

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter={"userId": {"$eq": user_id}}
    )

    if not results.matches:
        return "No relevant history found for your question.", []

    context_blocks = []
    for match in results.matches:
        m = match.metadata
        context_blocks.append(
            f"From: {m['url']} (visited {m['timestamp']})\nContent: {m['text']}\nRelevance: {match.score:.2f}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant answering based on browser history.

User's question:
{query}

Browsing history context:
{context}

Instructions:
- Only use the context above
- If info is missing, say so
- Cite source URLs

Answer:"""

    response = model.generate_content(prompt)
    return response.text, results.matches

# ================================
# Streamlit UI
# ================================
st.title("🧠 Smart Browser History")
st.markdown("Store, search, and ask questions over your browsing history using Gemini + Pinecone.")

# Section 1: Upload JSON Payload
st.header("📥 Upload JSON Payload")

sample_json = {
    "timestamp": datetime.now().isoformat() + "Z",
    "data": {
        "userId": "user123",
        "scrapedTextData": "This is the scraped text content from a webpage. " * 20,
        "url": "https://example.com/page"
    }
}

json_input = st.text_area("Paste JSON payload here:", value=json.dumps(sample_json, indent=2), height=300)

if st.button("🚀 Process Payload"):
    try:
        parsed = json.loads(json_input)
        result = process_json_payload(parsed)
        st.success(result)
    except Exception as e:
        st.error(f"❌ Invalid JSON: {str(e)}")

# Section 2: Q&A Interface
st.header("💬 Ask a Question")

with st.form("qa_form"):
    user_id = st.text_input("User ID", value="user123")
    user_query = st.text_area("What do you want to know?", height=100)
    submit = st.form_submit_button("Ask")

if submit and user_id and user_query:
    with st.spinner("Thinking..."):
        answer, matches = search_and_respond(user_query, user_id)
        st.subheader("🧠 Answer")
        st.write(answer)

        if matches:
            with st.expander("📚 Sources"):
                for m in matches:
                    meta = m.metadata
                    st.markdown(f"- [{meta['url']}]({meta['url']}) — Score: {m.score:.2f}")
