"""
Quiz generation via Gemini 1.5 Flash.
Uses strict JSON output format as specified.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json
import asyncio
import re
from typing import Any, Dict

from google import genai
from google.genai import types


GENERATION_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are an expert academic quiz generator.
Generate quiz strictly from the provided context.
Do not use outside knowledge.
If information is missing from the context, do not fabricate answers."""

MCQ_FORMAT = """{
  "type": "mcq",
  "questions": [
    {
      "question": "...",
      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "correct_answer": "A",
      "explanation": "..."
    }
  ]
}"""

DESCRIPTIVE_FORMAT = """{
  "type": "descriptive",
  "questions": [
    {
      "question": "...",
      "expected_points": ["point 1", "point 2", "point 3"]
    }
  ]
}"""

MIXED_FORMAT = """{
  "type": "mixed",
  "mcqs": [
    {
      "question": "...",
      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "correct_answer": "A",
      "explanation": "..."
    }
  ],
  "descriptive": [
    {
      "question": "...",
      "expected_points": ["point 1", "point 2"]
    }
  ]
}"""

FORMAT_MAP = {
    "mcq": MCQ_FORMAT,
    "descriptive": DESCRIPTIVE_FORMAT,
    "mixed": MIXED_FORMAT,
}


def _build_prompt(context: str, num_questions: int, difficulty: str, question_type: str, rubric: Dict[str, Any] = None) -> str:
    fmt = FORMAT_MAP[question_type]

    if question_type == "mixed":
        mcq_count = num_questions // 2
        desc_count = num_questions - mcq_count
        type_instruction = (
            f"Generate {mcq_count} MCQ questions and {desc_count} descriptive questions."
        )
    else:
        type_instruction = f"Generate exactly {num_questions} {question_type.upper()} questions."

    prompt = f"""Context (use ONLY this content to generate the quiz):
---
{context}
---

Requirements:
- {type_instruction}
- Difficulty: {difficulty}
- Question Type: {question_type.upper()}

Rules:
- For MCQ: Provide 4 options (A, B, C, D), one correct_answer letter, and a short explanation.
- For descriptive: Provide a list of expected_points (key ideas the answer should cover).
- Do NOT mention "the context" or "the passage" in the questions.
- Do NOT fabricate information not present in the context.
- Return STRICT JSON ONLY matching this exact format (no markdown, no preamble):

{fmt}"""

    # Append rubric instructions if provided
    if rubric:
      try:
        time_min = rubric.get('time_min')
        time_max = rubric.get('time_max')
        bloom = rubric.get('bloom')
        partial = rubric.get('partial_credit')
        points = rubric.get('points')
        rubric_lines = []
        if time_min is not None or time_max is not None:
          rubric_lines.append(f"Target solve time: {time_min}-{time_max} minutes")
        if bloom:
          rubric_lines.append(f"Bloom levels: {', '.join(bloom)}")
        rubric_lines.append(f"Partial credit allowed: {'yes' if partial else 'no'}")
        rubric_lines.append(f"Point value per question: {points}")
        prompt += "\nRubric:\n" + "\n".join(["- " + l for l in rubric_lines]) + "\n"
      except Exception:
        # ignore rubric formatting errors — prompt still works without it
        pass

    return prompt

def _extract_json(text: str) -> Dict[str, Any]:
    """Extract and parse JSON from model output, stripping any markdown fences."""
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = text.rstrip("`").strip()
    return json.loads(text)


async def generate_quiz(
  context: str,
  num_questions: int,
  difficulty: str,
  question_type: str,
  rubric: Dict[str, Any] = None,
  max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Call Gemini 1.5 Flash to generate a structured quiz from the retrieved context.
    Retries up to max_retries times on JSON parse failure.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)
    prompt = _build_prompt(context, num_questions, difficulty, question_type, rubric)
    loop = asyncio.get_event_loop()

    for attempt in range(max_retries + 1):
        raw = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3,
                    max_output_tokens=4096,
                ),
            )
        )

        text = raw.text or ""
        try:
            return _extract_json(text)
        except (json.JSONDecodeError, ValueError):
            if attempt == max_retries:
                raise ValueError(
                    f"Gemini returned invalid JSON after {max_retries + 1} attempts. "
                    f"Raw output:\n{text[:500]}"
                )
            # Retry on next iteration
            continue