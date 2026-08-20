"""Answer generation via Groq's LPU inference — chosen specifically for the
200ms-adjacent latency target; Groq's hosted open models return far faster
than typical LLM API latency. Retries + typed output for the harness
requirement."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

SYSTEM_PROMPT = (
    "You are a focused question-answering assistant. Answer ONLY using the "
    "provided context. If the context does not contain enough information to "
    "answer confidently, say so explicitly instead of guessing. Keep answers "
    "concise — 1-3 sentences unless the question requires more. "
    "IMPORTANT: Always respond in the SAME language and script as the user's "
    "question. If the question is in Hindi (Devanagari script), answer in Hindi. "
    "If the question is in Marathi (Devanagari script), answer in Marathi. "
    "If the question is in English, answer in English. Do not translate the "
    "question into English before answering — match the user's language exactly."
)


@dataclass
class GenerationResult:
    answer: str
    model: str
    raw_response: dict


class GenerationError(Exception):
    pass


class GroqGenerator:
    def __init__(self, api_key: Optional[str] = None, model: str = "openai/gpt-oss-20b",
                 timeout_seconds: float = 10.0):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key required — pass api_key= or set GROQ_API_KEY env var. "
                "Get one at https://console.groq.com/"
            )
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._client = None  # lazy init so import doesn't require network/SDK check

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self.api_key, timeout=self.timeout_seconds)
        return self._client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4), reraise=True)
    def generate(self, query: str, context_chunks: list[str]) -> GenerationResult:
        if not context_chunks:
            context_block = "(no context retrieved)"
        else:
            context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))

        user_prompt = (
            f"Context:\n{context_block}\n\nQuestion: {query}\n\n"
            f"(Remember: respond in the same language as the question above.)\n\nAnswer:"
        )

        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,  # low temp — factual QA, not creative writing
                max_tokens=300,
            )
        except Exception as e:
            raise GenerationError(f"Groq generation failed: {e}") from e

        answer = response.choices[0].message.content
        return GenerationResult(
            answer=answer,
            model=self.model,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else {},
        )
