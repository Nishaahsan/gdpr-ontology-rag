"""Evaluation harness for the GDPR chatbot: plain RAG vs ontology + RAG vs ontology only.

Run from the project folder (gdpr-rag) with Ollama running and the index built:

    python evaluation/run_eval.py --config plain    --mode full
    python evaluation/run_eval.py --config hybrid   --mode full
    python evaluation/run_eval.py --config ontology --mode full
    python evaluation/run_eval.py --compare          # prints a Markdown table to paste into the README

--mode retrieval is fast (no LLM): it only checks which articles were retrieved.
Results are saved in evaluation/results/<config>.json.

Metrics
  hit@k        : at least one expected article was retrieved / used
  recall@k     : fraction of the expected articles that were retrieved (matters for multi-article questions)
  MRR          : 1 / rank of the first retrieved chunk from an expected article
  fact_cov     : fraction of key facts that appear in the answer
  cite_ok      : the answer cites at least one expected article
  refusal      : out-of-scope questions get "does not cover this"; answerable questions do not
"""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "rag_core.py").exists() else HERE.parent     # works inside evaluation/ or at the repo root
sys.path.insert(0, str(ROOT))

QUESTIONS = next((p for p in (HERE / "eval_questions.json", ROOT / "evaluation" / "eval_questions.json") if p.exists()),
                 HERE / "eval_questions.json")
RESULTS = HERE / "results"
ABSTAIN_PHRASE = "does not cover this"


# ----------------------------------------------------------------------------- pure helpers (no Ollama needed)
def squash(s: str) -> str:
    """Lower-case and drop all whitespace so '4 %' matches '4%' and non-breaking spaces do not matter."""
    return re.sub(r"\s+", "", s.replace("\xa0", " ")).lower()


def retrieval_scores(expected, retrieved_articles):
    """expected: list[int]; retrieved_articles: list[int] in rank order (0 = recital chunk)."""
    exp = set(expected)
    got = set(retrieved_articles)
    hit = bool(exp & got)
    recall = len(exp & got) / len(exp) if exp else None
    rank = next((i + 1 for i, a in enumerate(retrieved_articles) if a in exp), None)
    return {"hit": hit, "recall": recall, "rr": (1 / rank) if rank else 0.0}


def cited_articles(answer: str):
    nums = re.findall(r"\[\s*Art\.?\s*(\d{1,3})", answer, flags=re.I)
    nums += re.findall(r"\bArticle\s+(\d{1,3})\b", answer, flags=re.I)
    return sorted({int(n) for n in nums})


def answer_scores(q, answer: str):
    abstained = ABSTAIN_PHRASE in answer.lower()
    if q["type"] == "out_of_scope":
        return {"abstain_ok": abstained, "fact_cov": None, "cite_ok": None, "cited": cited_articles(answer)}
    facts = q["key_facts"]
    cov = sum(squash(f) in squash(answer) for f in facts) / len(facts) if facts else None
    cited = cited_articles(answer)
    return {
        "abstain_ok": not abstained,
        "fact_cov": cov,
        "cite_ok": bool(set(cited) & set(q["expected_articles"])),
        "cited": cited,
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


# ----------------------------------------------------------------------------- running one configuration
def run_one(config, mode, k, col, onto, q):
    import rag_core as core
    t0 = time.time()
    facts, answer = [], None
    if config == "ontology":
        answer, docs, metas, facts = core.ontology_only_answer(col, onto, q["question"])
        retrieved = [f.article for f in facts if f.article]
    else:
        if config == "hybrid":
            docs, metas, facts = core.retrieve_hybrid(col, onto, q["question"], k)
        else:
            docs, metas = core.retrieve_plain(col, q["question"], k)
        retrieved = [m["article"] for m in metas]
        if mode == "full":
            answer = core.chat(core.build_messages(q["question"], docs, metas, facts if config == "hybrid" else None))

    row = {"id": q["id"], "type": q["type"], "question": q["question"],
           "expected": q["expected_articles"], "retrieved": retrieved}
    if q["type"] != "out_of_scope":
        row.update(retrieval_scores(q["expected_articles"], retrieved))
    if answer is not None:
        row["answer"] = answer
        row.update(answer_scores(q, answer))
    row["latency_s"] = round(time.time() - t0, 2)
    return row


def summarise(rows, config, mode, k):
    ans_rows = [r for r in rows if r["type"] != "out_of_scope"]
    oos_rows = [r for r in rows if r["type"] == "out_of_scope"]
    rel_rows = [r for r in rows if r["type"] == "relational"]
    s = {
        "name": config, "mode": mode, "k": k, "n_questions": len(rows),
        "hit@k": mean([float(r["hit"]) for r in ans_rows]),
        "recall@k": mean([r["recall"] for r in ans_rows]),
        "recall@k_relational": mean([r["recall"] for r in rel_rows]),
        "MRR": mean([r["rr"] for r in ans_rows]),
        "mean_latency_s": round(statistics.mean(r["latency_s"] for r in rows), 2),
    }
    if any("fact_cov" in r for r in rows):
        s.update({
            "fact_coverage": mean([r.get("fact_cov") for r in ans_rows]),
            "citation_ok": mean([float(r["cite_ok"]) for r in ans_rows if "cite_ok" in r]),
            "answered": mean([float(r["abstain_ok"]) for r in ans_rows if "abstain_ok" in r]),
            "out_of_scope_refused": mean([float(r["abstain_ok"]) for r in oos_rows if "abstain_ok" in r]),
        })
    return s


def compare():
    names = {"plain": "Plain RAG", "hybrid": "Ontology + RAG", "ontology": "Ontology only"}
    cols = [("hit@k", "hit@k"), ("recall@k", "recall@k"), ("recall@k_relational", "recall (list questions)"),
            ("fact_coverage", "fact coverage"), ("citation_ok", "citation OK"),
            ("answered", "answered"), ("out_of_scope_refused", "refused out-of-scope"),
            ("mean_latency_s", "latency (s)")]
    rows = []
    for key in ("plain", "hybrid", "ontology"):
        f = RESULTS / f"{key}.json"
        if f.exists():
            rows.append((names[key], json.loads(f.read_text(encoding="utf-8"))["summary"]))
    if not rows:
        print("No results found. Run run_eval.py for each --config first.")
        return
    print("| Configuration | " + " | ".join(c[1] for c in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for name, s in rows:
        print(f"| {name} | " + " | ".join(str(s.get(c[0], "-")) for c in cols) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["plain", "hybrid", "ontology"], default="plain")
    ap.add_argument("--mode", choices=["retrieval", "full"], default="full")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--compare", action="store_true", help="print a comparison table from saved results")
    args = ap.parse_args()
    if args.compare:
        return compare()

    import rag_core as core
    col = core.get_collection()
    onto = core.load_ontology() if args.config != "plain" else None
    if args.config != "plain" and onto is None:
        sys.exit("No ontology found. Put gdpr.owl in the ontology/ folder.")
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    rows = []
    for q in questions:
        row = run_one(args.config, args.mode, args.k, col, onto, q)
        rows.append(row)
        flag = "" if row.get("hit", True) else "  <-- expected article NOT retrieved"
        print(f"{q['id']} [{q['type']:<12}] articles {row['retrieved']} expected {q['expected_articles']}{flag}")

    summary = summarise(rows, args.config, args.mode, args.k)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.config}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== SUMMARY ===")
    for key, v in summary.items():
        print(f"{key:>22}: {v}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
