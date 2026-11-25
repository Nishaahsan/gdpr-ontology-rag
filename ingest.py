"""Step 1 - Ingest: split the GDPR file in docs/ into article-wise chunks and store them in ChromaDB.

Run:  python ingest.py
"""
from pathlib import Path

import chromadb
import ollama
from pypdf import PdfReader

from chunker import chunk_gdpr

DOCS_DIR = Path("docs")
DB_DIR = "chroma_db"
EMBED_MODEL = "nomic-embed-text"
BATCH = 32
MIN_CHARS = 5000          # the real GDPR text is ~350,000 characters; anything tiny is a placeholder/empty file


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def main():
    files = [p for p in DOCS_DIR.glob("*") if p.suffix.lower() in {".pdf", ".txt", ".md"}]
    if not files:
        print("The docs/ folder is empty. Put the GDPR text file (gdpr.txt or gdpr.pdf) in it.")
        return

    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection("gdpr")      # fresh index every run
    except Exception:
        pass
    col = client.create_collection("gdpr", metadata={"hnsw:space": "cosine"})

    total = 0
    for f in files:
        text = read_file(f)
        if len(text.strip()) < MIN_CHARS:
            print(f"SKIPPED {f.name}: only {len(text.strip())} characters. The file is empty or just a placeholder.")
            print("  If this should be the GDPR text: paste it into the file and press Ctrl+S to save, then run again.")
            continue
        chunks = chunk_gdpr(text)
        arts = sorted({c.article for c in chunks if c.article})
        print(f"{f.name}: {len(chunks)} chunks, {len(arts)} articles found")
        if len(arts) < 99:
            missing = [n for n in range(1, 100) if n not in arts]
            print(f"  WARNING: {len(missing)} of 99 articles missing: {missing[:20]}")
            print("  Check the file format (see 'Troubleshooting' in README.md).")
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i:i + BATCH]
            emb = ollama.embed(model=EMBED_MODEL, input=[c.text for c in batch])["embeddings"]
            col.add(
                ids=[f"{f.stem}-{total + j}" for j in range(len(batch))],
                documents=[c.text for c in batch],
                embeddings=emb,
                metadatas=[{"article": c.article or 0, "label": c.label, "title": c.title,
                            "part": c.part, "source": f.name} for c in batch],
            )
            total += len(batch)
            print(f"  embedded {min(i + BATCH, len(chunks))}/{len(chunks)}")
    if total == 0:
        print("Nothing was indexed. Fix the file above and run again.")
    else:
        print(f"Done: {total} chunks indexed. Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
