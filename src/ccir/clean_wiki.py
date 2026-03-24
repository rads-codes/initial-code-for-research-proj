import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LANGS = ["en", "de", "ro", "el"]

MAX_WORDS = 20

# basic multilingual-ish sentence splitter
SENT_SPLIT_RE = re.compile(r'(?<=[.!?;])\s+')

def split_sentences(text: str):
    parts = SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]

def chunk_to_max_words(sentence: str, max_words: int):
    words = sentence.split()
    if len(words) <= max_words:
        return [" ".join(words)]

    chunks = []
    for i in range(0, len(words), max_words):
        chunk = words[i:i + max_words]
        if len(chunk) >= 5:  # avoid tiny fragments
            chunks.append(" ".join(chunk))
    return chunks

def process_lang(lang: str) -> None:
    input_dir = REPO_ROOT / "extracted" / lang
    output_dir = REPO_ROOT / "data" / "raw" / "tmp" / "wiki_extracted" / lang
    output_file = output_dir / f"{lang}_sentences.jsonl"

    if not input_dir.exists():
        print(f"Skip {lang}: missing input dir {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    wiki_files = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.name.startswith("wiki_")
    )

    if not wiki_files:
        print(f"Skip {lang}: no wiki_* files found under {input_dir}")
        return

    file_count = 0
    row_count = 0
    skipped = 0

    with output_file.open("w", encoding="utf-8") as out:
        for file_path in wiki_files:
            file_count += 1
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue

                    text = obj.get("text", "").strip()
                    if not text:
                        skipped += 1
                        continue

                    sentences = split_sentences(text)

                    for sent in sentences:
                        # split long sentences into <=20-word chunks
                        chunks = chunk_to_max_words(sent, MAX_WORDS)

                        for chunk in chunks:
                            if len(chunk.split()) < 5:
                                continue

                            out.write(json.dumps({
                                "id": f"{lang}_{row_count}",
                                "text": chunk
                            }, ensure_ascii=False) + "\n")

                            row_count += 1

    print(
        f"{lang}: wrote {row_count} sentence-chunks from {file_count} files "
        f"(skipped {skipped}) -> {output_file}"
    )


def main():
    for lang in LANGS:
        process_lang(lang)


if __name__ == "__main__":
    main()