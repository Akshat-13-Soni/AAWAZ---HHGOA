"""Main pipeline orchestrator. This is the harness required by spec item #5:
structured input/output (Pydantic schemas), retries (delegated to each
component: SarvamSTT and GroqGenerator each retry their own external calls),
and error recovery (every stage is wrapped so one failing component degrades
gracefully into a typed refusal response instead of throwing a raw 500).

Every stage's wall-clock time is recorded into QueryResponse.stage_timings —
this doubles as the instrumentation the benchmark script (step 9) reads from.
"""
from __future__ import annotations

import base64
import time
from contextlib import contextmanager
from typing import Optional

from app.generation.groq_generator import GenerationError, GroqGenerator
from app.guardrails.groundedness import GroundednessChecker
from app.guardrails.input_gate import GateVerdict, InputGate
from app.harness.schemas import QueryRequest, QueryResponse, RetrievedContext, StageTiming
from app.retrieval.hybrid_retriever import HybridRetriever
from app.stt.sarvam_stt import SarvamSTT, SarvamSTTError


class RagOrchestrator:
    def __init__(
        self,
        retriever: HybridRetriever,
        generator: GroqGenerator,
        input_gate: InputGate,
        groundedness_checker: GroundednessChecker,
        stt: Optional[SarvamSTT] = None,
        top_k: int = 5,
    ):
        self.retriever = retriever
        self.generator = generator
        self.input_gate = input_gate
        self.groundedness_checker = groundedness_checker
        self.stt = stt
        self.top_k = top_k

    def handle_query(self, request: QueryRequest) -> QueryResponse:
        timings: list[StageTiming] = []
        t_start = time.perf_counter()

        # --- Stage 1: STT (only if audio was sent instead of text) ---
        query_text = request.text_query
        if query_text is None and request.audio_base64:
            with self._timed(timings, "stt"):
                if self.stt is None:
                    return self._refuse(timings, t_start, "STT not configured and no text_query provided")
                try:
                    audio_bytes = base64.b64decode(request.audio_base64)
                    transcription = self.stt.transcribe(audio_bytes, language_code=request.language_code)
                    query_text = transcription.text
                except SarvamSTTError as e:
                    return self._refuse(timings, t_start, f"transcription failed: {e}")

        if not query_text:
            return self._refuse(timings, t_start, "no query text available (empty audio transcription or no input)")

        # --- Stage 2: input gate (now runs a real retrieval-floor check) ---
        with self._timed(timings, "input_gate"):
            gate_result = self.input_gate.check(query_text)
        if gate_result.verdict != GateVerdict.ON_TOPIC:
            return self._refuse(timings, t_start, f"input gate: {gate_result.verdict.value} ({gate_result.reason})",
                                transcribed_query=query_text)

        # --- Stage 3: retrieval — reuse what the gate already fetched ---
        with self._timed(timings, "retrieval"):
            retrieved = gate_result.retrieved

        if not retrieved:
            return self._refuse(timings, t_start, "no relevant context found in corpus",
                                transcribed_query=query_text)

        context_texts = [r.chunk.text for r in retrieved]

        # --- Stage 4: generation ---
        with self._timed(timings, "generation"):
            try:
                gen_result = self.generator.generate(query_text, context_texts)
            except GenerationError as e:
                return self._refuse(timings, t_start, f"generation failed: {e}", transcribed_query=query_text)

        # --- Stage 5: groundedness check ---
        with self._timed(timings, "groundedness_check"):
            grounding = self.groundedness_checker.check(gen_result.answer, context_texts)

        if not grounding.is_grounded:
            return self._refuse(
                timings, t_start,
                f"answer failed groundedness check ({grounding.reason}) — refusing rather than "
                f"risking a hallucinated response",
                transcribed_query=query_text,
                retrieved_context=[
                    RetrievedContext(chunk_id=r.chunk.id, text=r.chunk.text, strategy=r.chunk.strategy,
                                      rrf_score=r.rrf_score)
                    for r in retrieved
                ],
            )

        total_ms = (time.perf_counter() - t_start) * 1000
        return QueryResponse(
            answer=gen_result.answer,
            answered=True,
            transcribed_query=query_text,
            retrieved_context=[
                RetrievedContext(chunk_id=r.chunk.id, text=r.chunk.text, strategy=r.chunk.strategy,
                                  rrf_score=r.rrf_score)
                for r in retrieved
            ],
            stage_timings=timings,
            total_duration_ms=total_ms,
        )

    @staticmethod
    @contextmanager
    def _timed(timings: list[StageTiming], stage_name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - t0) * 1000
            timings.append(StageTiming(stage=stage_name, duration_ms=duration_ms))
    @staticmethod
    def _localized_no_info_message(query_text: Optional[str]) -> str:
        if not query_text:
            return "I don't have enough grounded information to answer that confidently."
        marathi_markers = ["आहे", "काय", "नाही", "आणि"]
        hindi_markers = ["है", "क्या", "नहीं", "और"]
        marathi_hits = sum(1 for w in marathi_markers if w in query_text)
        hindi_hits = sum(1 for w in hindi_markers if w in query_text)
        if marathi_hits > hindi_hits:
            return "मला याचे विश्वासार्ह उत्तर देण्यासाठी पुरेशी माहिती नाही."
        has_devanagari = any('\u0900' <= ch <= '\u097F' for ch in query_text)
        if has_devanagari:
            return "मेरे पास इसका भरोसेमंद उत्तर देने के लिए पर्याप्त जानकारी नहीं है।"
        return "I don't have enough grounded information to answer that confidently."
    @staticmethod
    def _refuse(timings: list[StageTiming], t_start: float, reason: str,
                transcribed_query: Optional[str] = None,
                retrieved_context: Optional[list[RetrievedContext]] = None) -> QueryResponse:
        total_ms = (time.perf_counter() - t_start) * 1000
        return QueryResponse(
            answer=RagOrchestrator._localized_no_info_message(transcribed_query),
            answered=False,
            refusal_reason=reason,
            transcribed_query=transcribed_query,
            retrieved_context=retrieved_context or [],
            stage_timings=timings,
            total_duration_ms=total_ms,
        )
