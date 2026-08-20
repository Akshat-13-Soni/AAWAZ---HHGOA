with open("passages_preview.txt", "r", encoding="utf-8") as f:
    lines = [l for l in f if "[en]" in l]

sample = lines[::40]

with open("topic_sample.txt", "w", encoding="utf-8") as f:
    for line in sample:
        f.write(line)

print(f"Wrote {len(sample)} sampled topics out of {len(lines)} English entries.")