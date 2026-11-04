# Model Card - P22 Abstractive vs Extractive Summariser

This project is a pipeline, not a new model. It combines the following
pre-trained foundation models. All licenses below were read from each
model's Hugging Face page on 4 November 2026.

## Models used

| Purpose | Model | License | Notes |
|---|---|---|---|
| Extractive summary | TextRank (sumy implementation) | n/a (algorithm; sumy is Apache-2.0) | no training data involved |
| Abstractive summary | `sshleifer/distilbart-cnn-6-6` | Apache-2.0 | distilled BART fine-tuned on CNN/DailyMail (news) |
| LLM summary (fallback) | `Qwen/Qwen2.5-0.5B-Instruct` | Apache-2.0 | used when no API key is set |
| LLM summary (optional) | Groq `llama-3.3-70b-versatile` / Gemini `gemini-2.0-flash` via API | provider terms | only when the user configures a key in secrets |
| Figure captions | `Salesforce/blip-image-captioning-base` | BSD-3-Clause | trained on web image-text pairs |
| Faithfulness guardrail | `cross-encoder/nli-deberta-v3-xsmall` | Apache-2.0 | NLI: contradiction / entailment / neutral |
| Latent-space view | `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 | sentence embeddings + PCA |
| TTS audio | gTTS (Google Translate TTS) | library MIT; service is Google's unofficial free endpoint | demo/educational use |
| Thumbnail (optional) | `stabilityai/sd-turbo` via HF Inference API | Stability AI Community License (research/non-commercial use) | PIL title card is the no-token fallback |

## Intended use

Educational demonstration of extractive vs abstractive summarisation and
faithfulness checking for a university continuous-assessment project. Not
intended for medical, legal or financial decision support.

## Training data, bias and ethical considerations

- DistilBART was fine-tuned on English news articles (CNN/DailyMail), so it
  performs best on news-style English text and may drop details from other
  genres or from non-English input.
- BLIP captions reflect web image-text biases and can mislabel people,
  places or culturally specific scenes; captions are shown to the user
  before being injected into the summary.
- The NLI guardrail is a measured faithfulness screen, not a guarantee:
  in the benchmark it caught fabricated sentences (entailment 0.0 on a
  made-up cost figure) and flagged a small LLM's paraphrase drift, but it
  also produced paraphrase false positives - a flag means 'unverified',
  not 'wrong', and threshold choice (0.5) trades false alarms against misses.
- Verified live on the hosted app (Nov 4, 2026): the NLI guardrail matches
  each summary sentence against single source sentences, so a faithful
  sentence that COMBINES facts from several source sentences can be falsely
  flagged (observed: a correct 4-fact combined sentence flagged, entailment
  0.001) while a genuinely fabricated claim in the same summary was
  correctly flagged. Treat flags as a review prompt, not a verdict.
- Benchmark articles are synthetic, AI-authored texts written for this
  project, avoiding copyrighted news content and personal data; their
  reference summaries are also AI-authored, not human gold references.
- Generated thumbnails may reflect Stable Diffusion's training biases;
  the app labels which backend produced each thumbnail.

## Measured behaviour

See `benchmark/RESULTS.md` for the real ROUGE/latency table and the
guardrail's flag counts on 10 articles, plus the documented failure case
in `README.md` (article 8, local LLM paraphrase drift).
