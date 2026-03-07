"""
AI Quiz Assistant - FastAPI entry point
Stateless RAG pipeline: Upload → Chunk → Embed → Retrieve → Generate Quiz
"""

import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from lib.file_parser import parse_file
from lib.rag import build_vector_store, retrieve_chunks
from lib.quiz_generator import generate_quiz

app = FastAPI(title="AI Quiz Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────

class GenerateRequest(BaseModel):
    vector_store: list[dict]   # [{content: str, embedding: list[float]}]
    num_questions: int
    difficulty: str            # Easy | Medium | Hard
    question_type: str         # mcq | descriptive | mixed


# ─────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Step 1-4 of RAG pipeline:
      - Parse file  (PDF / TXT / DOCX)
      - Clean text
      - Chunk text  (800 chars, 150 overlap)
      - Generate embeddings via Gemini
    Returns the in-memory vector store to the client.
    """
    MAX_BYTES = 5 * 1024 * 1024  # 5 MB

    content = await file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds 5 MB limit.")

    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in ("pdf", "txt", "docx"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only PDF, TXT, and DOCX are accepted."
        )

    text = parse_file(content, ext)
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No readable text found in file. Scanned/image PDFs are not supported."
        )

    vector_store = await build_vector_store(text)

    return {
        "filename": filename,
        "chunk_count": len(vector_store),
        "vector_store": vector_store,
    }


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    """
    Step 5-6 of RAG pipeline:
      - Embed query → cosine similarity → top-5 chunks
      - Send context to Gemini 1.5 Flash → structured quiz JSON
    """
    if not req.vector_store:
        raise HTTPException(status_code=400, detail="Empty vector store. Upload a document first.")

    if not 1 <= req.num_questions <= 20:
        raise HTTPException(status_code=400, detail="num_questions must be between 1 and 20.")

    if req.difficulty not in ("Easy", "Medium", "Hard"):
        raise HTTPException(status_code=400, detail="difficulty must be Easy, Medium, or Hard.")

    if req.question_type not in ("mcq", "descriptive", "mixed"):
        raise HTTPException(status_code=400, detail="question_type must be mcq, descriptive, or mixed.")

    context_text = await retrieve_chunks(
        vector_store=req.vector_store,
        query=f"Generate {req.difficulty} {req.question_type} quiz questions about the main topics",
        top_k=5,
    )

    quiz = await generate_quiz(
        context=context_text,
        num_questions=req.num_questions,
        difficulty=req.difficulty,
        question_type=req.question_type,
    )

    return quiz


# ─────────────────────────────────────────────
# Serve React SPA (after API routes so they take priority)
# ─────────────────────────────────────────────

app.mount("/", StaticFiles(directory="static", html=True), name="static")