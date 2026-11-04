# Benchmark results (real run of evaluate_rouge.py)

Articles and reference summaries are synthetic, AI-authored for this
project (no copyrighted sources, no human annotators).

- Articles: 10 from test_articles.json (domains: business, culture, education, environment, finance, health, politics, science, sports, technology)
- Length preset: short
- NLI guardrail: cross-encoder/nli-deberta-v3-xsmall (flag threshold 0.5)
- Machine: x86_64, Python 3.10.12, CPU-only

| Method | Backend | ROUGE-1 | ROUGE-2 | ROUGE-L | Avg latency (s) | Unsupported sentences |
|---|---|---|---|---|---|---|
| textrank | sumy TextRank | 0.325 | 0.151 | 0.270 | 0.2 | 0/50 |
| bart | sshleifer/distilbart-cnn-6-6 (float16) | 0.406 | 0.163 | 0.313 | 6.2 | 7/27 |
| llm | local Qwen/Qwen2.5-0.5B-Instruct | 0.375 | 0.148 | 0.265 | 15.4 | 14/38 |

ROUGE scores are F1 against the AI-authored reference summary in
test_articles.json. Unsupported-sentence counts come from the NLI
guardrail (summary sentences whose best-matching source sentence
scores entailment below the threshold). Flags are classifier
signals of potentially unsupported text, not proven hallucinations.
