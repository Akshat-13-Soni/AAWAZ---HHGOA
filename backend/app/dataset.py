"""
Dataset loading + preprocessing for ai4bharat/MSMARCO-XI.

CONFIRMED SCHEMA (via --probe against the real hinval.parquet file — the full
dataset is ~55.6GB across multi-language train shards, so this loads a single
language's ~450-500MB `validation` file instead; see _validation_file_url()):

Each row is one QUERY, not one passage:
    query_id: int
    query: query text in the target language (e.g. Hindi)
    Eng_Query: the same query in English
    Answer / Eng_Answer: reference answer, target-language / English
    target_lang: e.g. "hin_Deva"
    query_type: e.g. "DESCRIPTION"
    passages: dict with THREE keys, each a list of 10 items in parallel:
        - English_passages: source passages in English
        - Translated_passages: the SAME passages translated into target_lang
        - is_selected: which passage(s) actually ground the reference answer

Design decision: index target-language passages for MULTIPLE languages —
default Hindi + Marathi (Marathi is widely spoken in Goa, this hackathon's
host state, alongside Konkani which isn't in this dataset) — so the demo
isn't locked to one Indic language. English passages are pulled only ONCE,
from the first language processed, since every language file pairs the same
underlying English source passages with its own translation — loading
English from each additional language would just duplicate the same content
repeatedly. The is_selected gold flag is kept in metadata for reference (e.g.
benchmark test-set construction, sanity-checking retrieval against
known-correct passages).
"""
from __future__ import annotations

import hashlib
from itertools import islice
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator, Optional

# 2-letter language code -> validation filename prefix, confirmed against the
# actual file listing at huggingface.co/datasets/ai4bharat/MSMARCO-XI/tree/main/validation
_LANGUAGE_FILE_MAP = {
    "as": "asmval", "bn": "benval", "gu": "gujval", "hi": "hinval",
    "kn": "kanval", "ml": "malval", "mr": "marval", "ne": "nepval",
    "or": "orival", "pa": "panval", "sa": "sanval", "ta": "tamval",
    "te": "telval", "ur": "urdval",
}

_VALIDATION_BASE_URL = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation"
_PREVIEW_CORPUS_PATH = Path(__file__).resolve().parent.parent / "passages_preview.txt"


@dataclass
class Passage:
    """A single retrievable unit of source text before chunking."""
    id: str
    text: str
    language: Optional[str] = None
    query: Optional[str] = None  # the query this passage was paired with
    metadata: dict = field(default_factory=dict)


def _make_id(*parts: str) -> str:
    raw = "||".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _validation_file_url(language: str) -> str:
    if language not in _LANGUAGE_FILE_MAP:
        raise ValueError(
            f"Unknown language '{language}'. Choose from: {list(_LANGUAGE_FILE_MAP)}"
        )
    return f"{_VALIDATION_BASE_URL}/{_LANGUAGE_FILE_MAP[language]}.parquet"


def load_preview_corpus(max_records: Optional[int] = None) -> Iterator[Passage]:
    """Load the small checked-in corpus used for resource-constrained demos.

    Unlike ``load_msmarco_xi``, this avoids importing the Hugging Face datasets
    runtime or downloading a parquet shard at application startup. Each line
    in ``passages_preview.txt`` is a previously exported MSMARCO-XI passage.
    """
    if not _PREVIEW_CORPUS_PATH.exists():
        raise FileNotFoundError(f"Preview corpus missing: {_PREVIEW_CORPUS_PATH}")

    emitted = 0
    with _PREVIEW_CORPUS_PATH.open(encoding="utf-8") as preview_file:
        for line_number, raw_line in enumerate(preview_file, start=1):
            if max_records is not None and emitted >= max_records:
                break

            try:
                _, tagged_text = raw_line.rstrip().split(". [", 1)
                language, text = tagged_text.split("] ", 1)
            except ValueError:
                continue

            text = text.strip()
            if not text:
                continue

            yield Passage(
                id=f"preview-{line_number}",
                text=text,
                language=language,
                metadata={"source": "checked-in-preview-corpus"},
            )
            emitted += 1


def load_msmarco_xi(
    max_records: Optional[int] = None,
    languages: Optional[list[str]] = None,
) -> Iterator[Passage]:
    """
    Loads validation files for one or more languages from ai4bharat/MSMARCO-XI
    and flattens them into Passage objects.

    Args:
        max_records: caps the number of QUERY ROWS read PER LANGUAGE FILE, not
                     total passages. Each row yields up to 10 target-language
                     passages (+ 10 English passages, but only from the first
                     language in the list — see below).
        languages: list of 2-letter codes to load, e.g. ["hi", "mr"] for
                   Hindi + Marathi (a nice touch given this is a Goa
                   hackathon — Marathi is widely spoken there alongside
                   Konkani). Defaults to ["hi", "mr"].

                   English passages are included ONCE, from the FIRST
                   language's file only — each per-language file pairs the
                   same underlying English source passages with its own
                   translation, so pulling English from every language would
                   just duplicate the same content repeatedly. One pass
                   through English is enough regardless of how many
                   translation languages you add.
    """
    from datasets import load_dataset

    if languages is None:
        languages = ["hi", "mr"]

    for lang_idx, language in enumerate(languages):
        url = _validation_file_url(language)
        # Streaming avoids downloading and converting the entire ~462 MB
        # validation parquet file before applying max_records. This is
        # essential for constrained deployment environments, where the
        # in-memory conversion can cause the service to restart during boot.
        ds = load_dataset("parquet", data_files=url, split="train", streaming=True)
        rows = islice(ds, max_records) if max_records is not None else ds

        include_english_this_pass = (lang_idx == 0)

        for row in rows:
            query_id = row.get("query_id")
            query_local = row.get("query")
            query_en = row.get("Eng_Query")
            answer_local = row.get("Answer")
            answer_en = row.get("Eng_Answer")
            target_lang = row.get("target_lang") or language
            query_type = row.get("query_type")

            passages_field = row.get("passages") or {}
            translated_passages = passages_field.get("Translated_passages") or []
            english_passages = passages_field.get("English_passages") or []
            is_selected = passages_field.get("is_selected") or []

            shared_metadata = {
                "query_id": query_id,
                "query_en": query_en,
                "query_local": query_local,
                "answer_local": answer_local,
                "answer_en": answer_en,
                "query_type": query_type,
            }

            for i, passage_text in enumerate(translated_passages):
                if not passage_text:
                    continue
                selected_flag = is_selected[i] if i < len(is_selected) else None
                pid = _make_id(str(query_id), str(i), str(target_lang))
                yield Passage(
                    id=pid,
                    text=passage_text,
                    language=target_lang,
                    query=query_local or query_en,
                    metadata={**shared_metadata, "is_selected": selected_flag},
                )

            if include_english_this_pass:
                for i, passage_text in enumerate(english_passages):
                    if not passage_text:
                        continue
                    selected_flag = is_selected[i] if i < len(is_selected) else None
                    pid = _make_id(str(query_id), str(i), "en")
                    yield Passage(
                        id=pid,
                        text=passage_text,
                        language="en",
                        query=query_en or query_local,
                        metadata={**shared_metadata, "is_selected": selected_flag},
                    )


def probe_schema(language: str = "hi") -> None:
    """Prints the first record's structure — row-level fields and the
    passages sub-fields — so you can re-confirm the schema if you switch
    languages or the dataset gets updated."""
    from datasets import load_dataset

    url = _validation_file_url(language)
    ds = load_dataset("parquet", data_files=url, split="train")
    first = ds[0]
    print(f"Loaded {len(ds)} query rows from {language} validation file.")
    print("Row-level columns:", list(first.keys()))
    for k, v in first.items():
        if k == "passages":
            print(f"  passages: keys={list(v.keys())}, "
                  f"len(Translated_passages)={len(v.get('Translated_passages', []))}")
        else:
            print(f"  {k}: {str(v)[:150]}")


if __name__ == "__main__":
    import sys

    if "--probe" in sys.argv:
        probe_schema()
    else:
        passages = list(load_msmarco_xi(max_records=2))
        langs_used = sorted({p.language for p in passages})
        print(f"Generated {len(passages)} passages from 2 query rows per language "
              f"(languages: {langs_used}).")
        for p in passages[:5]:
            print(p.id, p.language, "|", p.text[:80])
