"""
FastAPI application — wires every module built in steps 2-9 into one running
service. This is where placeholders become real: real embedder, real indexed
corpus, real Sarvam/Groq clients.

Run locally:
    export SARVAM_API_KEY=...
    export GROQ_API_KEY=...
    uvicorn app.main:app --reload --port 8000

Then point frontend/index.html's API_ENDPOINT at http://localhost:8000/api/query
"""
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 

load_dotenv()

from app.dataset import load_msmarco_xi, load_preview_corpus
from app.chunking import (
    FixedSizeChunker, SemanticChunker, SentenceWindowChunker, MetadataAwareChunker,
)
from app.generation.groq_generator import GroqGenerator
from app.guardrails.groundedness import GroundednessChecker
from app.guardrails.input_gate import InputGate, compute_corpus_centroid
from app.harness.orchestrator import RagOrchestrator
from app.harness.schemas import QueryRequest, QueryResponse
from app.retrieval.embedder import Embedder, HashEmbedder
from app.retrieval.hybrid_retriever import HybridRetriever
from app.stt.sarvam_stt import SarvamSTT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hhgoa_rag")

app = FastAPI(title="HH Goa 2026 — AAWAZ (Voice RAG)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your actual frontend origin before final submission
    allow_methods=["*"],
    allow_headers=["*"],
)

# Populated in the startup event below; module-level so the /api/query handler
# can reach it without re-threading it through FastAPI's dependency system for
# what is, for a hackathon demo, a single global pipeline instance.
orchestrator: RagOrchestrator | None = None

# How many MSMARCO-XI passages to index at startup. Kept small deliberately —
# see PROGRESS.md WARNING re: full dataset size / indexing time. Raise this
# only after confirming latency numbers still hold with headroom to spare.
MAX_PASSAGES = int(os.environ.get("MAX_PASSAGES", "2000"))
CORPUS_SOURCE = os.environ.get("CORPUS_SOURCE", "msmarco").lower()
EMBEDDER_MODE = os.environ.get("EMBEDDER_MODE", "sentence_transformer").lower()


@app.on_event("startup")
def build_pipeline():
    global orchestrator
    if CORPUS_SOURCE == "preview":
        logger.info(f"Loading up to {MAX_PASSAGES} passages from the preview corpus...")
        passages = list(load_preview_corpus(max_records=MAX_PASSAGES))
    elif CORPUS_SOURCE == "msmarco":
        logger.info(f"Loading up to {MAX_PASSAGES} passages from MSMARCO-XI...")
        passages = list(load_msmarco_xi(max_records=MAX_PASSAGES))
    else:
        raise ValueError("CORPUS_SOURCE must be 'preview' or 'msmarco'")
    logger.info(f"Loaded {len(passages)} passages.")

    if EMBEDDER_MODE == "hash":
        logger.info("Loading lightweight hash embedder for the demo...")
        embedder = HashEmbedder()
    elif EMBEDDER_MODE == "sentence_transformer":
        logger.info("Loading embedder (multilingual-e5-small)...")
        embedder = Embedder("intfloat/multilingual-e5-small")
    else:
        raise ValueError("EMBEDDER_MODE must be 'hash' or 'sentence_transformer'")

    logger.info("Chunking with all 4 strategies...")
    chunks = []
    chunks += list(FixedSizeChunker(chunk_size_tokens=128, overlap_tokens=24).chunk(passages))
    chunks += list(SemanticChunker(embedder=embedder).chunk(passages))
    chunks += list(SentenceWindowChunker(window_size=2).chunk(passages))
    chunks += list(MetadataAwareChunker(chunk_size_tokens=128, overlap_tokens=24).chunk(passages))
    logger.info(f"Produced {len(chunks)} chunks across 4 strategies.")

    logger.info("Indexing chunks (BM25 + FAISS)...")
    retriever = HybridRetriever(embedder=embedder)
    retriever.index(chunks)
    logger.info(f"Indexed {len(retriever)} chunks.")

    logger.info("Computing corpus centroid for the input gate...")
    sample_texts = [c.text for c in chunks[:500]]
    centroid = compute_corpus_centroid(embedder, sample_texts)

    # NOTE: these thresholds are UNTUNED defaults — see PROGRESS.md step 7.
    # Calibrate against real on/off-topic test queries before submission.
    input_gate = InputGate(embedder=embedder, corpus_centroid=centroid, off_topic_threshold=0.15)
    groundedness_checker = GroundednessChecker(embedder=embedder, min_lexical_overlap=0.15,
                                                min_semantic_similarity=0.35)

    generator = GroqGenerator()  # reads GROQ_API_KEY from env

    stt = None
    if os.environ.get("SARVAM_API_KEY"):
        stt = SarvamSTT()  # reads SARVAM_API_KEY from env
    else:
        logger.warning("SARVAM_API_KEY not set — voice input will be disabled, "
                        "text_query still works for testing.")

    orchestrator = RagOrchestrator(
        retriever=retriever,
        generator=generator,
        input_gate=input_gate,
        groundedness_checker=groundedness_checker,
        stt=stt,
        top_k=5,
    )
    logger.info("Pipeline ready.")


@app.get("/health")
def health():
    return {"status": "ok", "pipeline_ready": orchestrator is not None}


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if orchestrator is None:
        return QueryResponse(
            answer="Pipeline is still starting up — try again in a moment.",
            answered=False,
            refusal_reason="orchestrator not yet initialized",
        )
    return orchestrator.handle_query(request)
