# 🧠 MemExLens Server

MemExLens is an intelligent browser history search and Q&A system powered by Gemini (Google Generative AI) and Pinecone vector database. It enables you to store, search, and ask questions about your browsing history using advanced embeddings and retrieval-augmented generation.

---

## 🚀 Features

- **Store Browsing History**: Ingest and embed scraped webpage text, storing it as semantic vectors in Pinecone.
- **Semantic Search & Q&A**: Ask natural language questions about your browsing history and get answers grounded in your actual visited content.
- **User Isolation**: All data and queries are isolated per user.
- **FastAPI & Flask APIs**: REST endpoints for ingestion and Q&A.
- **Streamlit UI**: Interactive web app for testing, uploading, and querying.
- **Robust Environment Management**: All credentials and configs via `.env`.
- **Cloud Ready**: Dockerized and deployable to Google Cloud Run.

---

## 🏗️ Project Structure

```
MemExLens_Server/
│
├── VectorAgent/
│   ├── vector.py                # FastAPI server (Gemini + Pinecone)
│   ├── embedding_service.py     # Embedding & storage logic (Flask API)
│   ├── qa_service.py            # Q&A logic (Flask API)
│   ├── test_gemini.py           # Gemini API debug script
│   ├── test_pinecone.py         # Pinecone v7 test script
│   ├── test_geminy_pineconeQA.py    # Streamlit Q&A prototype
│   ├── test_geminy_pineconeQA2.py   # Streamlit JSON payload demo
│   └── delete_pinecone.py       # Utility to clear Pinecone index
│
├── app.py                       # Flask API server (production)
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker build file
├── .env.example                 # Example environment config
├── test_api.py                  # API test script
├── .gitignore, .dockerignore    # Ignore rules
└── README.md                    # This file
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/MemExLens_Server.git
cd MemExLens_Server
```

### 2. Create and Configure `.env`

Copy the example and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` and set:

- `GEMINI_API_KEY` (from [Google MakerSuite](https://makersuite.google.com/app/apikey))
- `PINECONE_API_KEY` (from [Pinecone Console](https://app.pinecone.io/))
- `PINECONE_ENVIRONMENT` (e.g., `gcp-starter` or `us-central1`)
- (Optional) Adjust chunking and logging configs

### 3. Install Python Dependencies

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 4. (Optional) Test Gemini & Pinecone Connectivity

```bash
python VectorAgent/test_gemini.py
python VectorAgent/test_pinecone.py
```

---

## 🖥️ Running the Servers

### A. **Flask API (Production/Cloud Run)**

```bash
python app.py
```

- Runs on `http://localhost:8080` by default.

### B. **FastAPI Server (Advanced, with OpenAPI docs)**

```bash
python VectorAgent/vector.py
```

- Runs on `http://localhost:8000`
- Interactive docs at `/docs`

### C. **Streamlit UI (Local Demo)**

```bash
streamlit run VectorAgent/test_geminy_pineconeQA.py
```

or

```bash
streamlit run VectorAgent/test_geminy_pineconeQA2.py
```

---

## 🧪 API Endpoints

### **Flask API (`app.py`)**

#### `POST /api/data`

Store a new browsing history entry.

**Payload:**
```json
{
  "timestamp": "2024-06-01T12:00:00Z",
  "data": {
    "userId": "user123",
    "scrapedTextData": "Full text of the webpage...",
    "url": "https://example.com/page"
  }
}
```

#### `POST /api/data/user`

Ask a question about a user's browsing history.

**Payload:**
```json
{
  "userId": "user123",
  "prompt": "What did I read about Python?"
}
```

#### `GET /health`

Health check endpoint.

---

### **FastAPI (`vector.py`)**

#### `POST /process-website`

Same as `/api/data` above.

#### `POST /search`

Same as `/api/data/user` above.

#### `GET /user/{user_id}/stats`

Get stats for a user.

#### `GET /health`, `GET /config`

Service health and config info.

---

## 📝 Environment Variables

All sensitive info is loaded from `.env`. Example:

```ini
GEMINI_API_KEY=your-gemini-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=gcp-starter
PINECONE_INDEX_NAME=browser-history-gemini
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_BATCH_SIZE=100
LOG_LEVEL=INFO
```

---

## 🐳 Docker & Cloud Run

### Build and Run Locally

```bash
docker build -t memexlens-server .
docker run -p 8080:8080 --env-file .env memexlens-server
```

### Deploy to Google Cloud Run

- See `.github/workflows/deploy.yaml` for CI/CD automation.
- Requires GCP project, Artifact Registry, and service account setup.

---

## 🧰 Utilities

- `test_api.py`: CLI script to test API endpoints.
- `delete_pinecone.py`: Delete all vectors from Pinecone index.
- `test_gemini.py`: Debug Gemini API connectivity.
- `test_pinecone.py`: Debug Pinecone v7 connectivity.

---

## 🧑‍💻 Development Notes

- **Embeddings**: Uses Gemini's `models/embedding-001` (768-dim).
- **Vector DB**: Pinecone v7 (Serverless, GCP region).
- **Chunking**: Configurable chunk size and overlap for long texts.
- **User Isolation**: All vectors are tagged with `userId`.
- **Security**: Never commit your real `.env` file.

---

## 📚 Example Usage

1. **Store a webpage** via `/api/data` or Streamlit UI.
2. **Ask a question** about your browsing history.
3. **Get answers** with cited sources and links.

---

## 📝 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙋 FAQ

- **Q:** My API key doesn't work!
  - **A:** Check `.env`, ensure no whitespace, and that your key is active.
- **Q:** Pinecone index not found?
  - **A:** The service will auto-create it if missing.
- **Q:** Can I use this for multiple users?
  - **A:** Yes, all data is isolated by `userId`.

---

## 🤝 Contributing

Pull requests welcome! Please open issues for bugs or feature requests.

---

## 📬 Contact

For questions, reach out via GitHub Issues or email the maintainer.

