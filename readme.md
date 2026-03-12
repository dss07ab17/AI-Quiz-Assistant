# 🎓 QuizCraft — AI Quiz Assistant

An AI-powered, stateless quiz generator built with **FastAPI** and **Gemini 2.5 Flash**.  
Upload any document → configure your quiz → get instant questions backed strictly by your content.

🔗 **Live Demo**: [quizcraft.vercel.app](https://quiz-craft-amber-tau.vercel.app)  
📦 **GitHub**: [github.com/dss07ab17/AI-Quiz-Assistant](https://github.com/dss07ab17/AI-Quiz-Assistant)

---

## ✨ Features

- 📄 Upload **PDF, TXT, or DOCX** files (up to 5 MB)
- 🧠 **RAG Pipeline** — questions generated strictly from your document
- 🎯 **MCQ, Descriptive, or Mixed** question formats
- 🎚️ **Easy, Medium, Hard** difficulty levels
- 🔢 Choose **1–20 questions** per quiz
- ⬇️ **Download quiz as PDF** (clean, print-ready)
- 🔒 **No database, no auth, no storage** — fully stateless
- ⚡ Powered by **Gemini 2.5 Flash** + **Gemini Embeddings**

---

## 🏗️ Architecture
```
Document Upload (PDF / TXT / DOCX)
          │
          ▼
┌─────────────────────────────────────────────────────┐
│                  FastAPI Backend                     │
│                                                      │
│  1. Text Extraction  (pypdf / mammoth / native)      │
│  2. Text Cleaning & Normalization                    │
│  3. Chunking         (800 chars, 150 overlap)        │
│  4. Gemini Embeddings → In-Memory Vector Store       │
│               [returned to browser]                  │
│                                                      │
│  5. User configures quiz (n, difficulty, type)       │
│  6. Query Embedding → Cosine Similarity Search       │
│  7. Top-5 Chunks → Context                          │
│  8. Gemini 2.5 Flash → Structured Quiz JSON          │
└─────────────────────────────────────────────────────┘
          │
          ▼
React Frontend (React 18 + Tailwind CSS)
  ├── MCQ cards with reveal toggle
  ├── Descriptive cards with expandable key points
  └── Download quiz as clean PDF
```

---

## 🧠 How RAG Works

**Retrieval-Augmented Generation (RAG)** ensures every question comes from your document — not the AI's general knowledge.

| Step | What happens |
|------|-------------|
| **1. Chunking** | Document split into 800-char overlapping windows |
| **2. Embedding** | Each chunk converted to a vector via `text-embedding-004` |
| **3. Retrieval** | Quiz request embedded → cosine similarity → top 5 chunks selected |
| **4. Generation** | Only those 5 chunks sent to Gemini 2.5 Flash as context |
| **5. Output** | Strict JSON quiz — MCQ or descriptive, no hallucination |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI (Python) |
| AI Model | Gemini 2.5 Flash |
| Embeddings | text-embedding-004 |
| RAG | In-memory cosine similarity |
| PDF Parse | pypdf |
| DOCX Parse | mammoth |
| Frontend | React 18 + Tailwind CSS (CDN) |
| PDF Export | jsPDF (CDN) |
| Deployment | Vercel |

---

## 📁 Project Structure
```
quiz-assistant/
├── main.py               # FastAPI app + API routes
├── api/
│   └── index.py          # Vercel entry point
├── lib/
│   ├── __init__.py
│   ├── file_parser.py    # PDF / DOCX / TXT extraction
│   ├── rag.py            # Chunking, embeddings, cosine similarity
│   └── quiz_generator.py # Gemini quiz generation
├── static/
│   └── index.html        # React SPA (no build step)
├── requirements.txt
├── vercel.json
├── .env.example
└── README.md
```

---

## 🚀 Local Setup
```bash
# 1. Clone the repo
git clone https://github.com/dss07ab17/AI-Quiz-Assistant.git
cd AI-Quiz-Assistant

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Add your Gemini API key
cp .env.example .env
# Open .env and set: GEMINI_API_KEY=your_key_here

# 6. Run the app
uvicorn main:app --reload --port 8000
```

Visit: **http://localhost:8000**

> Get a free Gemini API key at: https://aistudio.google.com/apikey

---

## ☁️ Deploy to Vercel

1. Push repo to GitHub
2. Go to [vercel.com](https://vercel.com) → **New Project** → Import repo
3. Add environment variable:
   - `GEMINI_API_KEY` = your Gemini API key
4. Click **Deploy**

> ⚠️ **Vercel Hobby Plan** has a 10-second function timeout. The RAG pipeline
> takes 15–30 seconds, so you may hit this limit. Consider upgrading to
> **Vercel Pro** (60s timeout) or deploy to
> [Railway](https://railway.app) / [Render](https://render.com) for free
> unlimited timeouts.

---

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Your Google Gemini API key |

Create a `.env` file based on `.env.example`:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## 🚧 Future Improvements

- ⏱️ Streaming generation for faster perceived response
- 📊 Score tracking — answer MCQs and see your final score
- 🗂️ Multi-document support — merge multiple files
- 🔍 Source highlighting — show which chunk each question came from
- 🌐 Share quiz via link
- 🖨️ Print-optimized PDF layout
- 🌍 Multi-language support

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<p align="center">Built with ❤️ using FastAPI · Gemini · RAG</p>