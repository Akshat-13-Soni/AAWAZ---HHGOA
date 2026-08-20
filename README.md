# Ask Anything — Voice-Enabled RAG (HH Goa 2026, Task 2)

Speak a question, get a grounded answer. Full pipeline: voice input → speech-to-text
(Sarvam) → multi-strategy chunking + hybrid retrieval (BM25 + dense, FAISS) →
guardrailed generation (Groq).

## Architecture

```
Mic input (browser)
   -> STT (Sarvam AI)
   -> Input gate (off-topic / unsafe rejection)
   -> Hybrid retrieval (BM25 + dense embeddings, RRF fusion,
      across 4 chunking strategies: fixed-size, semantic,
      sentence-window, metadata-aware)
   -> Generation (Groq / Llama 3.1 8B Instant)
   -> Groundedness check (refuse if the answer isn't supported
      by retrieved context)
   -> Response, with per-stage latency reported honestly
```

Every stage is timed independently — see `benchmarks/`. The 200ms target in
the task brief is realistically scoped to the chunking+retrieval portion of
the pipeline; STT and LLM generation involve real third-party API round trips
that are reported transparently alongside it rather than hidden inside one
misleading aggregate number.

## Project structure

```
backend/app/
  dataset.py        # MSMARCO-XI loading + preprocessing
  chunking/          # 4 chunking strategies behind one interface
  retrieval/         # embedder, FAISS store, BM25, hybrid RRF retriever
  stt/               # Sarvam speech-to-text client
  generation/        # Groq generation client
  guardrails/        # input gate + groundedness checker
  harness/           # orchestrator wiring everything with typed I/O + timing
  main.py            # FastAPI app / entrypoint
benchmarks/           # latency benchmark script (P50/P70/P100)
frontend/index.html   # hhgoa-themed voice UI with live pipeline visualization
docs/PROGRESS.md      # build log — what's done, what's untested, what to check
```

## Setup

```bash
cd backend
pip install -r requirements.txt --break-system-packages
cp .env.example .env   # fill in SARVAM_API_KEY and GROQ_API_KEY
```

Confirm the MSMARCO-XI schema once before relying on `dataset.py`:
```bash
python -m app.dataset --probe
```

Run the API:
```bash
uvicorn app.main:app --reload --port 8000
```

Open `frontend/index.html` in a browser (or serve it statically) with
`API_ENDPOINT` in the `<script>` pointed at your running backend.

## Before you submit — read `docs/PROGRESS.md`

It lists every module, what's been tested vs. only logic-verified, and three
things that specifically need your attention with real network access:

1. Confirm MSMARCO-XI's real column names (`--probe` above)
2. Calibrate guardrail thresholds against real on/off-topic test queries
3. Fire one real Sarvam STT call and confirm the response field name

## Latency benchmarking

```bash
python -m benchmarks.run_latency_benchmark --queries benchmarks/test_queries.txt
```

You'll need to fill in the real orchestrator construction at the marked spot
in that script — it deliberately refuses to run with mocked components so you
can't accidentally submit fabricated numbers.
