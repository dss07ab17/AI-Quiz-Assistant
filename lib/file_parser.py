"""
File text extraction:
  - PDF  → pypdf
  - DOCX → mammoth
  - TXT  → native decode
"""

import io
import re


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable chars."""
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_file(content: bytes, ext: str) -> str:
    """
    Extract raw text from uploaded file bytes.
    ext: 'pdf' | 'txt' | 'docx'
    """
    if ext == "txt":
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1", errors="replace")
        return _clean_text(text)

    elif ext == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return _clean_text("\n\n".join(pages))

    elif ext == "docx":
        import mammoth
        result = mammoth.extract_raw_text(io.BytesIO(content))
        return _clean_text(result.value)

    raise ValueError(f"Unsupported extension: {ext}")