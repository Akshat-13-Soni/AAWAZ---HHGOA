import os
from dotenv import load_dotenv
load_dotenv()
from app.dataset import load_msmarco_xi

n = int(os.environ.get("MAX_PASSAGES", 100))
passages = list(load_msmarco_xi(max_records=n))

with open("passages_preview.txt", "w", encoding="utf-8") as f:
    for i, p in enumerate(passages):
        lang = getattr(p, "language", "?")
        f.write(f"{i+1}. [{lang}] {p.text[:120]}...\n")

print(f"Wrote {len(passages)} passages to passages_preview.txt")