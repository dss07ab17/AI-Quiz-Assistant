# 🎓 AI Quiz Assistant

A stateless, production-ready RAG-powered quiz generator. Upload a document (PDF, TXT, DOCX), configure your quiz preferences, and get an AI-generated quiz backed strictly by the document content — no hallucination.

---

## Architecture
```
Document Upload
      │
      ▼
┌─────────────────────────────────────────────────┐
│                 FastAPI Backend                  │
│                                                  │
│  1. Text Extraction (pypdf / mammoth / native)   │
│  2. Text Cleaning & Normalization                │
│  3. Chunking (800 chars, 150 overlap)            │
│  4. Gemini Embeddings → In-Memory Vector Store   │
│              [returned to client]                │
│                                                  │
│  5. User configures quiz (n, difficulty, type)   │
│  6. Query Embedding → Cosine Similarity Search   │
│  7. Top-5 Chunks → Context                       │
│  8. Gemini 1.5 Flash → Structured Quiz JSON      │
└─────────────────────────────────────────────────┘
      │
      ▼
React Frontend (React 18, Tailwind CSS)
  - MCQ cards with reveal toggle
  - Descriptive cards with expandable key points
```

---

## Tech Stack

| Layer      | Choice                        |
|------------|-------------------------------|
| Backend    | FastAPI (Python)              |
| AI Model   | Gemini 1.5 Flash              |
| Embeddings | gemini-embedding-001          |
| RAG        | In-memory cosine similarity   |
| PDF Parse  | pypdf                         |
| DOCX Parse | mammoth                       |
| Frontend   | React 18 + Tailwind CSS (CDN) |
| Deploy     | Vercel                        |

---

## Local Setup
```bash
# 1. Clone the repo
git clone https://github.com/yourname/quiz-assistant
cd quiz-assistant

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variable
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 5. Run locally
uvicorn main:app --reload --port 8000
```

Visit: http://localhost:8000

---

## Deploy to Vercel

1. Push this repo to GitHub.
2. Go to [vercel.com](https://vercel.com) → **New Project** → Import repo.
3. In **Environment Variables**, add `GEMINI_API_KEY`.
4. Click **Deploy**.

> ⚠️ **Important**: The RAG pipeline (embedding + generation) takes 10–30 seconds.
> Vercel **Hobby** plan has a 10-second function timeout — you may hit this limit.
> **Vercel Pro** plan allows 60-second timeouts, which is sufficient.
> For Hobby plan, consider deploying to [Railway](https://railway.app) or [Render](https://render.com) instead.

---

## Project Structure
```
quiz-assistant/
├── main.py               # FastAPI app + API routes
├── lib/
│   ├── __init__.py
│   ├── file_parser.py    # PDF / DOCX / TXT text extraction
│   ├── rag.py            # Chunking, embeddings, cosine similarity
│   └── quiz_generator.py # Gemini quiz generation
├── static/
│   └── index.html        # React SPA (no build step needed)
├── requirements.txt
├── vercel.json
├── .env.example
└── README.md
```

---

## RAG Pipeline Explanation

**Retrieval-Augmented Generation (RAG)** grounds the AI's output in your document:

1. **Chunking** — the document is split into overlapping 800-character windows so no important context is cut off at boundaries.
2. **Embeddings** — each chunk is converted to a 3072-dimensional vector by `gemini-embedding-001`. These vectors capture semantic meaning.
3. **Retrieval** — when you click Generate, your quiz request is also embedded, then scored against all chunk embeddings using cosine similarity. The top 5 most relevant chunks are selected.
4. **Generation** — only those 5 chunks are sent as context to Gemini 1.5 Flash. The model is strictly instructed not to use outside knowledge.

This means every question is traceable back to your document.

---

## Future Improvements

- Streaming quiz generation for faster perceived response
- Export quiz as PDF or JSON
- Score tracking across sessions (using Vercel KV)
- Support for scanned PDFs via OCR
- Multi-document support
- Configurable chunk size and top-k retrieval