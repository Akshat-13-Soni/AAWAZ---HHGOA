path = "app/guardrails/groundedness.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old_content_words_fn = '''def _content_words(text: str) -> set[str]:
    # crude but language-agnostic-ish: strip punctuation, lowercase, drop very
    # short tokens (articles/particles contribute noise, not signal)
    tokens = re.findall(r"\\w+", text.lower())
    return {t for t in tokens if len(t) > 2}'''

new_content_words_fn = '''def _content_words(text: str) -> set[str]:
    # crude but language-agnostic-ish: strip punctuation, lowercase, drop very
    # short tokens (articles/particles contribute noise, not signal)
    tokens = re.findall(r"\\w+", text.lower())
    return {t for t in tokens if len(t) > 2}


# Phrases the LLM itself uses when it can't answer from context (see
# groq_generator.py's SYSTEM_PROMPT, which explicitly instructs it to say so
# rather than guess). If the generated *answer* itself matches one of these
# patterns, it's a refusal wearing an answer's clothes -- lexical/semantic
# overlap against the retrieved context is meaningless here, since an
# apology that mentions the topic ("I don't have info about pesticides...")
# will still share real content words with context about pesticides. This
# check runs BEFORE the overlap scoring and short-circuits to ungrounded,
# closing that loophole rather than trusting the overlap score to catch it.
_REFUSAL_PATTERNS = [
    # English
    r"i don'?t have (enough|sufficient) (grounded )?information",
    r"i don'?t have (that|this) info",
    r"i'?m sorry,? (but )?i don'?t have",
    r"the (provided |given )?context does not contain",
    r"cannot answer (that |this )?confidently",
    # Hindi (Devanagari)
    r"मुझे (पर्याप्त|भरोसेमंद) जानकारी नहीं",
    r"मेरे पास .{0,20}जानकारी नहीं",
    r"मुझे खेद है",
    r"संदर्भ में .{0,20}जानकारी नहीं",
    # Marathi (Devanagari)
    r"मला .{0,20}माहिती नाही",
    r"मला क्षमस्व",
    r"पुरेशी माहिती नाही",
    r"दिलेल्या संदर्भात .{0,20}नाही",
]

_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def _looks_like_refusal(answer: str) -> bool:
    """True if the answer text itself is an apology/refusal rather than a
    real attempt to answer -- see _REFUSAL_PATTERNS above for why this must
    be checked before, not instead of, the overlap scoring."""
    return bool(_REFUSAL_RE.search(answer))'''

old_check_start = '''    def check(self, answer: str, context_texts: list[str]) -> GroundednessResult:
        if not context_texts:
            return GroundednessResult(
                is_grounded=False, lexical_overlap=0.0, semantic_similarity=0.0,
                reason="no retrieved context to ground against",
            )

        combined_context = " ".join(context_texts)'''

new_check_start = '''    def check(self, answer: str, context_texts: list[str]) -> GroundednessResult:
        if not context_texts:
            return GroundednessResult(
                is_grounded=False, lexical_overlap=0.0, semantic_similarity=0.0,
                reason="no retrieved context to ground against",
            )

        if _looks_like_refusal(answer):
            return GroundednessResult(
                is_grounded=False, lexical_overlap=0.0, semantic_similarity=0.0,
                reason="generated answer is itself a refusal/apology, not an "
                       "attempted answer -- treating as ungrounded rather than "
                       "scoring lexical/semantic overlap against it",
            )

        combined_context = " ".join(context_texts)'''

if "_looks_like_refusal" in content:
    print("Already patched -- no changes made. (Both _content_words and "
          "check() already contain the refusal-detection code.)")
else:
    assert old_content_words_fn in content, "Could not find _content_words function -- aborting."
    assert old_check_start in content, "Could not find check() method start -- aborting."

    content = content.replace(old_content_words_fn, new_content_words_fn)
    content = content.replace(old_check_start, new_check_start)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print("Patched successfully: groundedness.py now detects refusal-shaped "
          "answers (EN/HI/MR) and short-circuits to ungrounded before scoring "
          "lexical/semantic overlap.")
