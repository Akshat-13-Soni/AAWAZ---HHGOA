import re

path = "app/harness/orchestrator.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old_refuse_sig = '''    @staticmethod
    def _refuse(timings: list[StageTiming], t_start: float, reason: str,
                transcribed_query: Optional[str] = None) -> QueryResponse:
        total_ms = (time.perf_counter() - t_start) * 1000
        return QueryResponse(
            answer=RagOrchestrator._localized_no_info_message(transcribed_query),
            answered=False,
            refusal_reason=reason,
            transcribed_query=transcribed_query,
            stage_timings=timings,
            total_duration_ms=total_ms,
        )'''

new_refuse_sig = '''    @staticmethod
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
        )'''

old_groundedness_call = '''        if not grounding.is_grounded:
            return self._refuse(
                timings, t_start,
                f"answer failed groundedness check ({grounding.reason}) \u2014 refusing rather than "
                f"risking a hallucinated response",
                transcribed_query=query_text,
            )'''

new_groundedness_call = '''        if not grounding.is_grounded:
            return self._refuse(
                timings, t_start,
                f"answer failed groundedness check ({grounding.reason}) \u2014 refusing rather than "
                f"risking a hallucinated response",
                transcribed_query=query_text,
                retrieved_context=[
                    RetrievedContext(chunk_id=r.chunk.id, text=r.chunk.text, strategy=r.chunk.strategy,
                                      rrf_score=r.rrf_score)
                    for r in retrieved
                ],
            )'''

assert old_refuse_sig in content, "Could not find _refuse method — aborting, no changes made."
assert old_groundedness_call in content, "Could not find groundedness refusal call — aborting, no changes made."

content = content.replace(old_refuse_sig, new_refuse_sig)
content = content.replace(old_groundedness_call, new_groundedness_call)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("Patched successfully: _refuse now accepts retrieved_context, and the "
      "groundedness-check refusal now passes it through.")
