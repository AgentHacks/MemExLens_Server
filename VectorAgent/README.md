# Browser History Search System

A semantic search system that indexes and queries browser history using embeddings, allowing users to find information from previously visited web pages using natural language queries.

## Overview

This system consists of two main services:
- **Embedding Service** (`embedding_service.py`) - Processes and stores web page content as vector embeddings
- **Q&A Service** (`qa_service.py`) - Enables natural language querying of stored browsing history

## Architecture

```
User's Browser History → Text Extraction → Chunking → Embedding → Vector Storage → Semantic Search
```

## Key Features

- **Semantic Search**: Find information from your browsing history using natural language questions
- **Chunk-based Processing**: Large documents are split into manageable chunks with overlap for better context
- **User Isolation**: Each user's data is stored separately and queries only return their own results
- **Temporal Context**: Results include timestamps showing when pages were visited
- **Efficient Storage**: Uses vector embeddings for fast similarity search

## Technologies Used

- **Google Gemini API**: For generating text embeddings and answering questions
- **Pinecone**: Vector database for storing and querying embeddings
- **Python**: Core programming language
- **dotenv**: Environment variable management

## How It Works

### 1. Embedding Service (`embedding_service.py`)

This service handles the ingestion and storage of web page content:

1. **Text Chunking**: 
   - Splits scraped web page text into chunks of 6000 characters
   - Uses 300 character overlap between chunks to maintain context
   - Skips chunks with less than 50 characters

2. **Embedding Generation**:
   - Uses Google's `embedding-001` model to convert text chunks into 768-dimensional vectors
   - Each chunk is embedded with its URL as the title for better context

3. **Vector Storage**:
   - Stores embeddings in Pinecone with metadata including:
     - User ID
     - Source URL
     - Timestamp
     - Text preview (first 1000 chars)
     - Chunk position information

### 2. Q&A Service (`qa_service.py`)

This service enables querying the stored browsing history:

1. **Query Processing**:
   - Converts user's natural language question into an embedding
   - Searches Pinecone for the top 10 most similar chunks from the user's history

2. **Context Assembly**:
   - Retrieves matching text chunks with their metadata
   - Formats timestamps into readable dates
   - Filters out low-quality matches

3. **Answer Generation**:
   - Uses Gemini 1.5 Flash model to generate comprehensive answers
   - Provides source attribution with links and visit timestamps
   - Returns formatted markdown with summary and visited links sections

## Setup

### Prerequisites

- Python 3.7+
- Google Cloud account with Gemini API access
- Pinecone account

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install google-generativeai pinecone-client python-dotenv
   ```

3. Create a `.env` file with the following variables:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   PINECONE_INDEX_NAME=browser-history-prototype
   ```

## Usage

### Storing Browser History

```python
from embedding_service import embed_and_store_in_pinecone

data = {
    "userId": "user123",
    "url": "https://example.com/article",
    "scrapedTextData": "Full text content of the web page..."
}

embed_and_store_in_pinecone(data, "2024-01-15T10:30:00Z")
```

### Querying Browser History

```python
from qa_service import generate_answer

answer = generate_answer(
    user_id="user123",
    prompt="What did I read about machine learning last week?"
)
print(answer)
```

### Output Format

The Q&A service returns markdown-formatted responses:

```markdown
# Summary

Based on your browsing history, you read several articles about machine learning...

# Visited Links

- [https://example.com/ml-basics](https://example.com/ml-basics) — *January 10, 2024 at 2:30 PM UTC*
- [https://blog.ai/neural-networks](https://blog.ai/neural-networks) — *January 12, 2024 at 9:15 AM UTC*
```

## Configuration

### Embedding Parameters

- `EMBED_DIM`: 768 (dimension of Gemini embeddings)
- `CHUNK_SIZE`: 6000 characters per chunk
- `CHUNK_OVERLAP`: 300 characters overlap between chunks

### Search Parameters

- `top_k`: Number of similar chunks to retrieve (default: 10)
- Model: Gemini 1.5 Flash for answer generation

## Privacy & Security

- User data is isolated by `userId` filtering
- Each user can only query their own browsing history
- Chunk IDs are generated using MD5 hashing for uniqueness

## Error Handling

- Failed embeddings default to zero vectors to prevent data loss
- Service continues processing even if individual chunks fail
- Q&A service returns graceful error messages if generation fails

## Limitations

- Maximum chunk size of 6000 characters may split important context
- Requires active internet connection for API calls
- Vector search may miss exact keyword matches
- Storage costs scale with browsing history volume

## Future Enhancements

- Add date range filtering for queries
- Implement incremental updates for changed pages
- Add support for multimedia content extraction
- Enable cross-user knowledge sharing (with permissions)
- Implement local caching for frequently accessed data

