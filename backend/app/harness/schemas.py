"""Typed request/response schemas shared across the harness. Pydantic models
give us the "structured input/output handling" requirement for free and catch
malformed data at the boundary rather than deep inside the pipeline."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    audio_base64: Optional[str] = Field(default=None, description="Base64-encoded audio for voice input")
    text_query: Optional[str] = Field(default=None, description="Direct text query, bypassing STT (for testing/fallback)")
    language_code: Optional[str] = Field(default=None, description="BCP-47 language hint, e.g. 'hi-IN'")


class StageTiming(BaseModel):
    stage: str
    duration_ms: float


class RetrievedContext(BaseModel):
    chunk_id: str
    text: str
    strategy: str
    rrf_score: float


class QueryResponse(BaseModel):
    answer: str
    answered: bool  # False if guardrails triggered a refusal
    refusal_reason: Optional[str] = None
    transcribed_query: Optional[str] = None
    retrieved_context: list[RetrievedContext] = Field(default_factory=list)
    stage_timings: list[StageTiming] = Field(default_factory=list)
    total_duration_ms: float = 0.0
