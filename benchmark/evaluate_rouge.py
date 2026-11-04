"""Benchmark: ROUGE-1/2/L + latency for the three summarisation methods.

Runs TextRank, BART/T5 and the LLM backend over benchmark/test_articles.json,
scores each summary against the article's AI-authored reference summary,
checks faithfulness with the Unit V NLI guardrail, and writes:

  benchmark/results.csv  - one row per article x method
  benchmark/RESULTS.md   - aggregated table for the report and README

The run is split into a summarise phase (one model family loaded at a
time) and a score phase (ROUGE + NLI), so it fits on small machines:

  python benchmark/evaluate_rouge.py --phase both            # full run
  python benchmark/evaluate_rouge.py --start 0 --limit 2 --append

Model choices are read from the environment (BART_MODEL, BART_DTYPE,
LLM_LOCAL_MODEL, LLM_DTYPE, NLI_MODEL) so the same script runs with small
models on a laptop and full-size models on the deployment host. Scores in
RESULTS.md must always come from a real run of this script - never edit
them by hand.
"""

import argparse
import csv
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARIES_JSON = os.path.join(HERE, "summaries.json")
RESULTS_CSV = os.path.join(HERE, "results.csv")
METHODS = ["textrank", "bart", "llm"]


def _free_models():
    """Drop cached models so the next family loads into free memory."""
    import core.u2_summarizers as u2
    import core.u5_guardrail as u5
    u2._seq2seq = None
    u2._seq2seq_tokenizer = None
    u2._causal = None
    u2._causal_tokenizer = None
    u5._nli = None
    gc.collect()


def summarise_one(method, article, length):
    from core.u2_summarizers import (LENGTH_TO_TOKENS, bart_summary,
                                     llm_summary, textrank_summary)
    from core.u1_prompting import build_prompt
    max_new, _ = LENGTH_TO_TOKENS.get(length, LENGTH_TO_TOKENS["medium"])
    if method == "textrank":
        return textrank_summary(article, sentence_count=5), "sumy TextRank"
    if method == "bart":
        return bart_summary(article, length=length), (
            os.environ.get("BART_MODEL", "sshleifer/distilbart-cnn-6-6")
            + f" ({os.environ.get('BART_DTYPE', 'float32')})")
    if method == "llm":
        return llm_summary(build_prompt(article, "role_format", length,
                                        "paragraph"),
                           max_tokens=max_new + 60)
    raise ValueError(method)


def phase_summarize(articles, methods, length):
    store = {}
    if os.path.exists(SUMMARIES_JSON):
        store = json.load(open(SUMMARIES_JSON))
    for method in methods:          # one model family in memory at a time
        _free_models()
        for a in articles:
            key = f"{a['id']}:{method}:{length}"
            if key in store:
                continue
            t0 = time.time()
            summary, backend = summarise_one(method, a["article"], length)
            store[key] = {"summary": summary, "backend": backend,
                          "latency_s": round(time.time() - t0, 2)}
            print(f"  summarised id={a['id']} {method} "
                  f"({store[key]['latency_s']}s)", flush=True)
            json.dump(store, open(SUMMARIES_JSON, "w"), indent=1)
    _free_models()
    return store


def phase_score(articles, methods, length, append):
    from rouge_score import rouge_scorer
    from core.u5_guardrail import check_faithfulness, hallucination_report
    store = json.load(open(SUMMARIES_JSON))
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                      use_stemmer=True)
    rows = []
    for a in articles:
        for method in methods:
            key = f"{a['id']}:{method}:{length}"
            if key not in store:
                print(f"  !! missing summary {key}, run summarise first")
                continue
            entry = store[key]
            scores = scorer.score(a["reference"], entry["summary"])
            verdicts = check_faithfulness(a["article"], entry["summary"])
            rep = hallucination_report(verdicts)
            rows.append({
                "id": a["id"], "domain": a["domain"], "method": method,
                "backend": entry["backend"],
                "rouge1": round(scores["rouge1"].fmeasure, 4),
                "rouge2": round(scores["rouge2"].fmeasure, 4),
                "rougeL": round(scores["rougeL"].fmeasure, 4),
                "latency_s": entry["latency_s"],
                "sentences": rep["sentences_checked"],
                "flagged": rep["sentences_flagged"],
                "summary": entry["summary"],
            })
            print(f"[{a['id']:>2}] {method:<9} R1={rows[-1]['rouge1']:.3f} "
                  f"R2={rows[-1]['rouge2']:.3f} RL={rows[-1]['rougeL']:.3f} "
                  f"{entry['latency_s']:5.1f}s flags={rep['sentences_flagged']}"
                  f"/{rep['sentences_checked']}", flush=True)

    if append and os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV) as f:
            prior = [r for r in csv.DictReader(f)
                     if not any(r["id"] == str(n["id"]) and
                                r["method"] == n["method"] for n in rows)]
        rows = prior + rows
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def write_results_md(rows, articles, length):
    agg = {}
    for r in rows:
        a2 = agg.setdefault(r["method"], {"n": 0, "r1": 0.0, "r2": 0.0,
                                          "rl": 0.0, "lat": 0.0, "flag": 0,
                                          "sent": 0, "backend": r["backend"]})
        a2["n"] += 1
        a2["r1"] += float(r["rouge1"])
        a2["r2"] += float(r["rouge2"])
        a2["rl"] += float(r["rougeL"])
        a2["lat"] += float(r["latency_s"])
        a2["flag"] += int(r["flagged"])
        a2["sent"] += int(r["sentences"])

    import platform
    lines = [
        "# Benchmark results (real run of evaluate_rouge.py)",
        "",
        "Articles and reference summaries are synthetic, AI-authored for this",
        "project (no copyrighted sources, no human annotators).",
        "",
        f"- Articles: {len(articles)} from test_articles.json "
        f"(domains: {', '.join(sorted({a['domain'] for a in articles}))})",
        f"- Length preset: {length}",
        f"- NLI guardrail: {os.environ.get('NLI_MODEL', 'cross-encoder/nli-deberta-v3-xsmall')}"
        f" (flag threshold {os.environ.get('NLI_THRESHOLD', '0.5')})",
        f"- Machine: {platform.machine()}, Python {platform.python_version()}, CPU-only",
        "",
        "| Method | Backend | ROUGE-1 | ROUGE-2 | ROUGE-L | Avg latency (s) | Unsupported sentences |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in METHODS:
        if m not in agg:
            continue
        a2 = agg[m]
        lines.append(
            f"| {m} | {a2['backend']} | {a2['r1']/a2['n']:.3f} | "
            f"{a2['r2']/a2['n']:.3f} | {a2['rl']/a2['n']:.3f} | "
            f"{a2['lat']/a2['n']:.1f} | {a2['flag']}/{a2['sent']} |")
    lines += [
        "",
        "ROUGE scores are F1 against the AI-authored reference summary in",
        "test_articles.json. Unsupported-sentence counts come from the NLI",
        "guardrail (summary sentences whose best-matching source sentence",
        "scores entailment below the threshold). Flags are classifier",
        "signals of potentially unsupported text, not proven hallucinations.",
    ]
    with open(os.path.join(HERE, "RESULTS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--start", type=int, default=0,
                    help="skip this many articles first (chunked runs)")
    ap.add_argument("--append", action="store_true",
                    help="merge with an existing results.csv")
    ap.add_argument("--length", default="medium")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--phase", choices=["summarize", "score", "both"],
                    default="both")
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    articles = json.load(open(os.path.join(HERE, "test_articles.json")))
    articles = articles[args.start: args.start + args.limit]

    if args.phase in ("summarize", "both"):
        phase_summarize(articles, methods, args.length)
    if args.phase in ("score", "both"):
        rows = phase_score(articles, methods, args.length, args.append)
        write_results_md(rows, articles, args.length)
        print("wrote results.csv and RESULTS.md")


if __name__ == "__main__":
    main()
