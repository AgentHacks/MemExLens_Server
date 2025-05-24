import os
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai
import hashlib
import logging
from datetime import datetime

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EMBED_DIM = 768
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "browser-history-prototype")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")

# Init Gemini & Pinecone
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")  # or gemini-pro
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


def get_gemini_embedding(text: str):
    embedding_response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return embedding_response['embedding']


def query_pinecone(user_id: str, prompt: str, top_k: int = 10):
    query_vector = get_gemini_embedding(prompt)
    filter_by_user = {
        "userId": {"$eq": user_id}
    }
    results = index.query(vector=query_vector, top_k=top_k, filter=filter_by_user, include_metadata=True)
    return results.matches

    from datetime import datetime

from datetime import datetime

def format_timestamp(iso_str):
    try:
        dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        return dt.strftime('%B %d, %Y at %I:%M %p UTC')
    except Exception:
        return iso_str


def generate_answer(user_id: str, prompt: str) -> str:
    try:
        matches = query_pinecone(user_id, prompt, top_k=15)

        if not matches:
            return "Summary:\nI couldn't find any relevant information from your history.\n\nVisited Links:\nNone"

        context_blocks = []
        visited_links_set = set()

        for match in matches:
            metadata = match.get('metadata', {})
            text = metadata.get('text', '')
            url = metadata.get('url', '')
            timestamp = metadata.get('timestamp', '')

            if text:
                readable_time = format_timestamp(timestamp) if timestamp else "Unknown time"
                # Only add URL if present
                if url:
                    visited_links_set.add((url, readable_time))
                    source_note = f"\n[Visited]({url}) on {readable_time}"
                else:
                    source_note = ""
                context_blocks.append(f"{text}{source_note}")

        context = "\n\n".join(context_blocks)

        full_prompt = f"""You are an intelligent assistant. Use the following browsing context to answer the user's question.

Browsing History Context:
{context}

User Question: {prompt}

Answer:"""

        response = model.generate_content(full_prompt)

        # Build final visited links list, sorted by timestamp if desired
        visited_links_sorted = sorted(visited_links_set, key=lambda x: x[1])
        visited_links_section = "\n".join([f"{ts} - {url}" for url, ts in visited_links_sorted]) if visited_links_sorted else "None"

        final_output = f"""Summary:
{response.text.strip()}

Visited Links:
{visited_links_section}
"""
        return final_output

    except Exception as e:
        logger.error(f"Error in Q&A generation: {e}")
        return "Summary:\nAn error occurred while trying to generate an answer.\n\nVisited Links:\nNone"





# For direct testing
if __name__ == "__main__":
    uid = input("Enter user ID: ")
    user_prompt = input("Enter your question: ")
    answer = generate_answer(uid, user_prompt)
    print("\nAnswer:\n", answer)
