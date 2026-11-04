# Abstractive vs Extractive Summariser (GenAI CA Mini Project P22)

Summarise an article **three ways** - TextRank (extractive), BART/T5
(encoder-decoder abstractive) and an LLM - then check which summaries stay
faithful with an NLI guardrail. Built for the Generative AI continuous
assessment mini project (Roll 22 → P22), covering syllabus units U1-U5 in
one deployable Gradio app.

**Live demo:** _to be added once deployed_
**Author:** Krish Kumar (Roll 22)

## Unit coverage

| Unit | Requirement | Where it lives |
|---|---|---|
| U1 Foundations | Length & style prompts; 2 prompt techniques compared; latent-space view | `core/u1_prompting.py`, sidebar settings, "Prompt-technique comparison" + "Latent-space view" sections |
| U2 GANs & LLMs | TextRank vs BART/T5 vs LLM; attention explanation (Track B deep layer) | `core/u2_summarizers.py`, cross-attention heatmap |
| U3 Multimodal | BLIP figure captions injected into the article so they enter the summary | `core/u3_multimodal.py`, figure upload |
| U4 Tools & generation | TTS audio summary + generated thumbnail | `core/u4_media.py` |
| U5 Responsible AI | NLI model flags unsupported (hallucinated) sentences; model card; licence note | `core/u5_guardrail.py`, `MODEL_CARD.md`, `LICENSE` |

Deliverables: summariser app ✓ · ROUGE table on 10 articles
(`benchmark/RESULTS.md`) ✓ · hallucination flags (in-app + benchmark) ✓

## Quickstart

```bash
pip install -r requirements.txt
python app.py        # serves on http://localhost:7860
```

No API keys are required: without keys the app runs its "LLM" method on a
small local open model (Qwen2.5-0.5B-Instruct), so every feature works on a
free CPU host.

### Optional API keys (Settings → Secrets on the host)

| Secret | Effect |
|---|---|
| `GROQ_API_KEY` | LLM method uses Groq (`GROQ_MODEL`, default llama-3.3-70b-versatile) |
| `GEMINI_API_KEY` | LLM method uses Gemini (`GEMINI_MODEL`, default gemini-2.0-flash) |
| `HF_TOKEN` | Thumbnail uses the HF Inference API (SD-Turbo) instead of the PIL title-card fallback |

Keys are read from the environment / Streamlit secrets only. **No keys are
stored in this repo.** Free tiers for these APIs exist but have provider
rate limits that change; the app falls back cleanly if a call fails.

### Deploying on Hugging Face Spaces (free)

The app uses Gradio because HF's native Streamlit SDK is deprecated and
Docker Spaces require a paid plan; free accounts can run Gradio Spaces on
ZeroGPU (CPU Basic, 2 vCPU / 16 GB).

1. Create a Space (SDK: Gradio).
2. Push this repo's files (or link the GitHub repo).
3. Add any optional secrets under Settings → Secrets
   (`GROQ_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN`).
4. The Space installs `requirements.txt` and runs `app.py`.

## Benchmark

```bash
python benchmark/evaluate_rouge.py            # full run, medium length
python benchmark/evaluate_rouge.py --length short
```

Real measured results (CPU-only run) are in
[benchmark/RESULTS.md](benchmark/RESULTS.md): per-method ROUGE-1/2/L F1
against AI-authored reference summaries, average latency, and the number
of sentences the NLI guardrail flagged as unsupported. `test_articles.json`
contains 10 synthetic articles across 10 domains, each with an AI-authored
reference summary; they were generated for this project and are not copied
from any news source. ROUGE against a single AI-authored reference
understates abstractive quality, so read the table as a relative
comparison, not an absolute quality score.

### Documented failure case

In the measured run, the local 0.5B LLM's summary of article 8 (night-shift
safety rules) had **all 5 sentences flagged** by the NLI guardrail. One is a
genuine factual drift: "a register of cab drivers verified by the police"
became "registration of cab drivers by law enforcement". Others are
paraphrases the NLI classifier could not match back to the source - false
positives, which is the guardrail's own known limitation: **a flag means
"could not be verified against the source", not "proven wrong"**, and
unflagged text is not guaranteed correct. Small local LLMs drift more;
using the Groq/Gemini backend reduces it. See `benchmark/results.csv`
row id=8, method=llm.

## Repository layout

```text
├── app.py                     # Streamlit app entry point
├── core/
│   ├── u1_prompting.py        # length/style prompts, prompt techniques, latent view
│   ├── u2_summarizers.py      # TextRank, BART/T5, LLM backends, attention heatmap
│   ├── u3_multimodal.py       # BLIP captioning + context injection
│   ├── u4_media.py            # gTTS audio + SD-Turbo/PIL thumbnail
│   └── u5_guardrail.py        # NLI faithfulness checker
├── benchmark/
│   ├── test_articles.json     # 10 original articles + reference summaries
│   ├── evaluate_rouge.py      # ROUGE-1/2/L + latency + NLI flag evaluation
│   └── RESULTS.md             # real measured results
├── assets/                    # architecture diagram, screenshots
├── MODEL_CARD.md
├── requirements.txt
└── LICENSE                    # MIT (project code only; models have their own licenses)
```

## Ethics and limitations

- The NLI guardrail reduces the risk of silently wrong summaries but is
  itself a model: it can miss hallucinations or flag correct paraphrases.
- Summaries of third-party articles remain subject to the source's
  copyright; summarise only text you may use.
- TTS uses Google's translate TTS endpoint via gTTS (free, unofficial);
  generated audio is for study/demo use.
- See MODEL_CARD.md for model licenses, bias notes and intended use.
