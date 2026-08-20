# Build Progress — HH Goa Task 2 (Voice RAG)

Deadline plan: code freeze Aug 20, videos Aug 21, buffer Aug 22.

## Status

| # | Step | Status | Notes |
|---|------|--------|-------|
| 1 | Repo scaffold | ✅ done | backend/frontend/benchmarks/docs structure created |
| 2 | Dataset loader + preprocessing (MSMARCO-XI) | ✅ done | Schema fields are best-guess — run `python -m app.dataset --probe` first with real internet access to confirm column names |
| 3 | Chunking strategies (4 + common interface) | ✅ done | fixed_size, semantic, sentence_window, metadata_aware — all under `backend/app/chunking/`. Tested fixed_size/sentence_window/metadata_aware directly; semantic needs an embedder, verified in step 4 |
| 4 | Vector DB + hybrid retrieval (BM25 + dense + RRF) | ✅ done | FAISS (in-process, HNSW) + BM25 + RRF fusion under `backend/app/retrieval/`. Tested end-to-end with a mock embedder (sandbox has no HF access) — indexing, fusion, and top-k correctness all verified. Swap `MockEmbedder` for the real `Embedder` class (bge-small) when running with internet access — no other code changes needed |
| 5 | STT module (Sarvam) | ✅ done | `backend/app/stt/sarvam_stt.py` — retries via tenacity, typed result. Import/error-path tested; could NOT verify live API response field names (no sandbox access to sarvam.ai) — fire one real test call early and adjust `payload.get("transcript"/"text")` if the real field name differs |
| 6 | Harness/orchestrator (typed I/O, retries, latency logging) | ✅ done | `backend/app/harness/` — Pydantic schemas + orchestrator wiring STT→gate→retrieval→generation→groundedness with per-stage timing. Tested 3 scenarios end-to-end (happy path, empty-input refusal, hallucination refusal) — all correct. Mocked retrieval+guardrail stages ran in ~1.3ms combined, confirming that layer isn't the latency risk — STT/Groq API round trips are |
| 7 | Guardrails (input gate + groundedness check) | ✅ done, needs threshold calibration | `backend/app/guardrails/`. Algorithm logic verified correct (unsafe-pattern block ✅, hallucination correctly rejected on lexical overlap ✅). Off-topic/semantic thresholds are UNTUNED — my mock embedder is too crude (bag-of-words hash) to validate real separation. Once running with real bge-small embedder: feed 5-10 known on/off-topic queries and tune `off_topic_threshold` (input_gate.py) and `min_semantic_similarity` (groundedness.py) against real numbers before trusting defaults |
| 8 | Generation module (Groq) | ✅ done | `backend/app/generation/groq_generator.py` — retries, typed result, low temp for factual QA. Import/instantiation tested; live Groq calls untested (no sandbox network access) |
| 9 | Latency benchmark script (P50/P70/P100) | ✅ done | `benchmarks/run_latency_benchmark.py` — percentile math + full aggregation tested against a 30-query mocked run, correct output. `main()` deliberately raises `NotImplementedError` at the real-orchestrator wiring point (not silently stubbed) so you can't accidentally submit fake numbers — fill in your real Sarvam/Groq/embedder-backed orchestrator there. Task requires 200ms full pipeline; use this to report per-stage numbers honestly (see notes above on STT/generation being the real bottleneck) |
| 10 | Frontend (hhgoa-themed, mic input) | ✅ built, NOT visually verified | `frontend/index.html` — self-contained, no build step. Design: night-jungle dark palette + phosphor-green/sunset-coral accents, Space Grotesk/Inter/JetBrains Mono. Signature element: live pipeline-stage tracker showing real per-stage latency as it runs (ties directly to the harness/latency requirements). Mic capture via MediaRecorder + Web Audio; Enter-key text-query fallback for demo safety. Calls `POST /api/query` matching the QueryRequest/QueryResponse schemas already built and tested. **Could not screenshot-render in sandbox (no headless browser, Chromium download domain not on network allowlist) — open it in an actual browser before trusting the visuals** |
| 11 | Wire end-to-end + deployment configs | ✅ done | `backend/app/main.py` — FastAPI app with real startup wiring (dataset load → embedder → 4-strategy chunking → hybrid index → guardrails → generator → orchestrator), `/health` and `/api/query` endpoints. **Tested via FastAPI TestClient with a mocked orchestrator injected** — real HTTP routing + exact response schema confirmed correct. The real `build_pipeline()` startup path (HF dataset + bge-small + Sarvam + Groq) is UNTESTED — needs real network access to run. Added `.env.example`, `.gitignore`, `railway.json`, top-level `README.md` |

## Key decisions locked in
- STT: Sarvam AI (Indic-language support fits MSMARCO-XI)
- Generation: Groq-hosted model (latency)
- Vector DB: FAISS (in-process, no network hop — best shot at the 200ms budget) with a Qdrant-compatible interface if you want to swap later
- Chunking: fixed-size+overlap, semantic, sentence-window, metadata-aware — combined via hybrid BM25+dense retrieval with reciprocal rank fusion
- Latency reporting: every stage timed and logged separately; the 200ms claim is scoped to chunking+retrieval, STT/generation timed and reported honestly alongside it

## How to resume
Read this file top to bottom, pick up at the first ⏳ row. Each step's code lives under `backend/app/<module>/`.

## Requires you to supply (not committed to repo)
- Sarvam API key
- Groq API key
- Actual MSMARCO-XI download (sandbox has no HF access — loader code is written against the `datasets` library and will pull it live when you run it with internet access)
