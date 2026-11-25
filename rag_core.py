"""Shared pipeline used by both the chatbot (app.py) and the evaluation (evaluation/run_eval.py).

Three configurations
  plain     : vector search + local LLM
  hybrid    : the ontology adds facts and extra articles to the vector search, then the local LLM answers
  ontology  : no LLM at all; the answer is built from ontology facts plus the source text of the articles
"""
import re
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
DB_DIR = str(ROOT / "chroma_db")
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"
NUM_CTX = 8192            # hybrid prompts are longer than the default context window
ABSTAIN_PHRASE = "does not cover this"
EXTRA_ARTICLES_MAX = 6    # how many ontology-suggested articles may be added to the vector results

SYSTEM_PLAIN = (
    "You are a GDPR study assistant. Answer ONLY from the CONTEXT below. "
    "Cite every claim with the article label exactly as shown, e.g. [Art. 17]. "
    "If the context does not contain the answer, reply exactly: "
    "\"The provided GDPR text does not cover this.\" "
    "Keep the answer under 180 words, in simple English. "
    "This is educational information, not legal advice."
)
SYSTEM_HYBRID = (
    "You are a GDPR study assistant. You get ONTOLOGY FACTS (structured knowledge) and CONTEXT (GDPR text). "
    "Answer ONLY from them. Cite every claim with the article label, e.g. [Art. 17]. "
    "For list questions, cover every item the ontology facts list. "
    "If neither contains the answer, reply exactly: \"The provided GDPR text does not cover this.\" "
    "Keep the answer under 200 words, in simple English. "
    "This is educational information, not legal advice."
)

CONFIGS = {
    "plain": "Plain RAG",
    "hybrid": "Ontology + RAG (hybrid)",
    "ontology": "Ontology only (no LLM)",
}


# ----------------------------------------------------------------------------- loading
def get_collection():
    import chromadb
    return chromadb.PersistentClient(path=DB_DIR).get_collection("gdpr")


def load_ontology():
    """Returns an Ontology, or None if no ontology/gdpr.owl exists."""
    from ontology import Ontology, find_ontology_file
    path = find_ontology_file(ROOT)
    return Ontology.load(path) if path else None


# ----------------------------------------------------------------------------- retrieval
def article_chunks(col, number: int, max_parts: Optional[int] = None):
    got = col.get(where={"article": int(number)})
    rows = sorted(zip(got["documents"], got["metadatas"]), key=lambda r: r[1].get("part", 0))
    return rows[:max_parts] if max_parts else rows


def retrieve_plain(col, question: str, k: int, direct_lookup: bool = True) -> Tuple[List[str], List[dict]]:
    import ollama
    docs, metas, seen = [], [], set()
    if direct_lookup:      # the user typed "Article 17": fetch that article first
        for num in re.findall(r"\b(?:article|art\.?)\s*(\d{1,3})\b", question, flags=re.I):
            for d, m in article_chunks(col, num):
                if d not in seen:
                    seen.add(d); docs.append(d); metas.append(m)
    emb = ollama.embed(model=EMBED_MODEL, input=question)["embeddings"][0]
    res = col.query(query_embeddings=[emb], n_results=k)
    for d, m in zip(res["documents"][0], res["metadatas"][0]):
        if d not in seen:
            seen.add(d); docs.append(d); metas.append(m)
    return docs[: k + 2], metas[: k + 2]


def retrieve_hybrid(col, onto, question: str, k: int):
    """Vector search + articles suggested by the ontology. Returns docs, metas, facts."""
    result = onto.query(question)
    docs, metas = retrieve_plain(col, question, k)
    have = {m["article"] for m in metas}
    extra_docs, extra_metas = [], []
    for art in result.articles:
        if art in have or len(extra_docs) >= EXTRA_ARTICLES_MAX:
            continue
        rows = article_chunks(col, art, max_parts=1)
        if rows:
            extra_docs.append(rows[0][0]); extra_metas.append(rows[0][1]); have.add(art)
    # ontology-suggested articles go first: they are the structured, precise evidence
    return extra_docs + docs, extra_metas + metas, result.facts


# ----------------------------------------------------------------------------- generation
def build_messages(question: str, docs, metas, facts=None):
    context = "\n\n".join(f"[{m['label']}] {d}" for d, m in zip(docs, metas))
    if facts:
        block = "\n".join(f"- {f.cited()}" for f in facts)
        user = f"ONTOLOGY FACTS:\n{block}\n\nCONTEXT:\n{context}\n\nQUESTION: {question}"
        return [{"role": "system", "content": SYSTEM_HYBRID}, {"role": "user", "content": user}]
    return [{"role": "system", "content": SYSTEM_PLAIN},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}]


def chat(messages, stream: bool = False):
    import ollama
    opts = {"temperature": 0, "num_ctx": NUM_CTX}
    if stream:
        return ollama.chat(model=LLM_MODEL, messages=messages, stream=True, options=opts)
    return ollama.chat(model=LLM_MODEL, messages=messages, options=opts)["message"]["content"]


def ontology_only_answer(col, onto, question: str):
    """No LLM. Returns (answer_text, docs, metas, facts)."""
    result = onto.query(question)
    if not result.facts:
        return "The ontology does not cover this. Try the hybrid mode.", [], [], []
    lines = ["Based on the GDPR ontology:", ""]
    lines += [f"- {f.cited()}" for f in result.facts]
    docs, metas = [], []
    for art in result.articles[:2]:                       # attach the official text of the first articles
        rows = article_chunks(col, art, max_parts=1)
        if rows:
            docs.append(rows[0][0]); metas.append(rows[0][1])
    if docs:
        lines += ["", "Source text:"]
        lines += [f"> [{m['label']}] {d[:350]}{'...' if len(d) > 350 else ''}" for d, m in zip(docs, metas)]
    return "\n".join(lines), docs, metas, result.facts
