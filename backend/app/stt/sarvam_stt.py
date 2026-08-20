"""Sarvam AI speech-to-text client. Chosen over ElevenLabs because MSMARCO-XI
is Indic-language data — Sarvam is purpose-built for Indian languages and
transcribes them meaningfully better than English/European-focused providers.

Wrapped with tenacity retries (requirement #5: harness with retry/error
recovery) and returns a typed result rather than raw JSON.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


@dataclass
class TranscriptionResult:
    text: str
    language_code: Optional[str]
    raw_response: dict


class SarvamSTTError(Exception):
    pass


class SarvamSTT:
    def __init__(self, api_key: Optional[str] = None, timeout_seconds: float = 15.0):
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Sarvam API key required — pass api_key= or set SARVAM_API_KEY env var. "
                "Get one at https://www.sarvam.ai/"
            )
        self.timeout_seconds = timeout_seconds

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        reraise=True,
    )
    def transcribe(self, audio_bytes: bytes, filename: str = "audio.webm",
                    content_type: str = "audio/webm",
                    language_code: Optional[str] = None) -> TranscriptionResult:
        """
        Args:
            audio_bytes: raw audio file bytes (wav/mp3 — check Sarvam docs for
                         the exact accepted formats/sample rates at call time).
            language_code: optional BCP-47-style code (e.g. "hi-IN") to hint the
                         language; omit to let Sarvam auto-detect.
        """
        headers = {"api-subscription-key": self.api_key}
        data = {}
        if language_code:
            data["language_code"] = language_code
        files = {"file": (filename, audio_bytes, content_type)}

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(SARVAM_STT_URL, headers=headers, data=data, files=files)

        if response.status_code != 200:
            raise SarvamSTTError(f"Sarvam STT failed: {response.status_code} {response.text}")

        payload = response.json()
        # NOTE: confirm exact response field names against current Sarvam API
        # docs at call time — using the documented "transcript" field as of
        # this writing, with a fallback.
        text = payload.get("transcript") or payload.get("text") or ""
        return TranscriptionResult(
            text=text,
            language_code=payload.get("language_code", language_code),
            raw_response=payload,
        )
