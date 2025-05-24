# 🔎 MemExLens Server

MemExLens is an intelligent browser history search and Q&A system powered by Gemini (Google Generative AI) and Pinecone vector database. It enables you to store, search, and ask questions about your browsing history using advanced embeddings and retrieval-augmented generation.

[Click here to navigate to UI repository](https://github.com/AgentHacks/MemExLens_UI)

---

## 🚀 Features

- **Store Browsing History**: Ingest and embed scraped webpage text, storing it as semantic vectors in Pinecone.
- **Semantic Search & Q&A**: Ask natural language questions about your browsing history and get answers grounded in your actual visited content.
- **User Isolation**: All data and queries are isolated per user.
- **FastAPI & Flask APIs**: REST endpoints for ingestion and Q&A.
- **Streamlit UI**: Interactive web app for testing, uploading, and querying.
- **Robust Environment Management**: All credentials and configs via `.env`.
- **Cloud Ready**: Dockerized and deployable to Google Cloud Run.
- **Chunk-based Processing**: Large documents are split into manageable chunks with overlap for better context.
- **Temporal Context**: Results include timestamps showing when pages were visited.
- **Efficient Storage**: Uses vector embeddings for fast similarity search.

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

## Demo

![Demo](demo1.gif)

![Screenshot](screenshot2.png)

## 🖼️ Architecture

![Architecture Diagram](arch.png)


**Description:**
- **Text Extraction**: Scraped text from visited web pages.
- **Chunking**: Splits large text into overlapping chunks for better context.
- **Gemini Embedding API**: Converts text chunks into semantic vectors (768-dim).
- **Pinecone Vector DB**: Stores vectors with metadata (user, URL, timestamp).
- **Semantic Search & Retrieval**: Finds relevant chunks for user queries.
- **Q&A Generation**: Gemini model generates answers grounded in retrieved content.

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

## 📚 Example Usage

### Storing Browser History (Python)

```python
from VectorAgent.embedding_service import embed_and_store_in_pinecone

data = {
    "userId": "user123",
    "url": "https://example.com/article",
    "scrapedTextData": "Full text content of the web page..."
}

embed_and_store_in_pinecone(data, "2024-01-15T10:30:00Z")
```

### Querying Browser History (Python)

```python
from VectorAgent.qa_service import generate_answer

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

---

## 🔧 Configuration

### Embedding Parameters

- `EMBED_DIM`: 768 (dimension of Gemini embeddings)
- `CHUNK_SIZE`: 6000 characters per chunk (default, configurable)
- `CHUNK_OVERLAP`: 300 characters overlap between chunks (default, configurable)

### Search Parameters

- `top_k`: Number of similar chunks to retrieve (default: 10)
- Model: Gemini 1.5 Flash for answer generation

---

## 🧰 Utilities

- `test_api.py`: CLI script to test API endpoints.
- `delete_pinecone.py`: Delete all vectors from Pinecone index.
- `test_gemini.py`: Debug Gemini API connectivity.
- `test_pinecone.py`: Debug Pinecone v7 connectivity.

---

## 🐳 Docker & Cloud Run

### Build and Run Locally

```bash
docker build -t memexlens-server .
docker run -p 8080:8080 --env-file .env memexlens-server
```

### Deploy to Google Cloud Run

You can deploy MemExLens Server to [Google Cloud Run](https://cloud.google.com/run) for scalable, serverless hosting. The project includes a GitHub Actions workflow for automated CI/CD.

#### **Manual Deployment Steps**

1. **Create a Google Cloud Project**  
   - Enable Cloud Run, Artifact Registry, and IAM APIs.

2. **Create Artifact Registry**  
   - Example:  
     ```bash
     gcloud artifacts repositories create memexlens-server --repository-format=docker --location=us-east1
     ```

3. **Build and Push Docker Image**  
   - Authenticate Docker with GCP:  
     ```bash
     gcloud auth configure-docker us-east1-docker.pkg.dev
     ```
   - Build and push:
     ```bash
     docker build -t us-east1-docker.pkg.dev/<PROJECT_ID>/memexlens-server/memexlens-server:latest .
     docker push us-east1-docker.pkg.dev/<PROJECT_ID>/memexlens-server/memexlens-server:latest
     ```

4. **Deploy to Cloud Run**  
   - Deploy the image:
     ```bash
     gcloud run deploy memexlens-server \
       --image us-east1-docker.pkg.dev/<PROJECT_ID>/memexlens-server/memexlens-server:latest \
       --region us-east1 \
       --set-env-vars GEMINI_API_KEY=... \
       --set-env-vars PINECONE_API_KEY=... \
       --set-env-vars PINECONE_ENVIRONMENT=... \
       --set-env-vars PINECONE_INDEX_NAME=...
     ```

5. **Access the Service**  
   - After deployment, Cloud Run will provide a public HTTPS URL.

---

## 🔄 CI/CD: GitHub Actions Workflow

This project uses a GitHub Actions workflow (`.github/workflows/deploy.yaml`) to automate deployment to Cloud Run on every push to the `main` branch.

### **Workflow Steps**

1. **Checkout Code**  
   - Uses `actions/checkout` to pull the latest code.

2. **Authenticate with Google Cloud**  
   - Uses `google-github-actions/auth` with a service account key stored in GitHub Secrets.

3. **Set Up Cloud SDK**  
   - Installs and configures the Google Cloud SDK.

4. **Configure Docker for Artifact Registry**  
   - Enables Docker to push images to GCP Artifact Registry.

5. **Build Docker Image**  
   - Builds the Docker image using the provided `Dockerfile`.

6. **Push Docker Image**  
   - Pushes the built image to Artifact Registry.

7. **Deploy to Cloud Run**  
   - Deploys the new image to Cloud Run, passing all required environment variables (API keys, config).

8. **Output Service URL**  
   - Prints the deployed Cloud Run service URL for reference.

### **Environment Variables & Secrets**

- `GEMINI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`, etc. are securely passed from GitHub Secrets/Variables.
- The workflow uses a service account with permissions for Cloud Run and Artifact Registry.

### **How to Use**

- Push to `main` branch triggers deployment.
- Update secrets and variables in your GitHub repository settings as needed.
- See `.github/workflows/deploy.yaml` for full details.

---

## 🔒 Privacy & Security

- User data is isolated by `userId` filtering.
- Each user can only query their own browsing history.
- Chunk IDs are generated using MD5 hashing for uniqueness.
- Never commit your real `.env` file.

---

## 🛠️ Error Handling

- Failed embeddings default to zero vectors to prevent data loss.
- Service continues processing even if individual chunks fail.
- Q&A service returns graceful error messages if generation fails.

---

## ⚠️ Limitations

- Maximum chunk size of 6000 characters may split important context.
- Requires active internet connection for API calls.
- Vector search may miss exact keyword matches.
- Storage costs scale with browsing history volume.

---

## 🚀 Future Enhancements

- Add date range filtering for queries.
- Implement incremental updates for changed pages.
- Add support for multimedia content extraction.
- Enable cross-user knowledge sharing (with permissions).
- Implement local caching for frequently accessed data.

---

## 🧑‍💻 Development Notes

- **Embeddings**: Uses Gemini's `models/embedding-001` (768-dim).
- **Vector DB**: Pinecone v7 (Serverless, GCP region).
- **Chunking**: Configurable chunk size and overlap for long texts.
- **User Isolation**: All vectors are tagged with `userId`.

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

For questions, reach out via GitHub Issues or email the maintainers [sarthakd.work@gmail.com](sarthakd.work@gmail.com), [kartikraut023@gmail.com](kartikraut023@gmail.com), [aadityakasbekar@gmail.com](aadityakasbekar@gmail.com)

## Contributors

- [aadityaKasbekar](https://github.com/aadityaKasbekar)
- [kartikraut98](https://github.com/kartikraut98)
- [sarthak-deshmukh1999](https://github.com/sarthak-deshmukh1999)

