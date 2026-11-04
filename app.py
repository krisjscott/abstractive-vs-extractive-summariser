"""P22 - Abstractive vs Extractive Summariser (Generative AI CA mini project).

Gradio app (HF Spaces' free tier runs Gradio on CPU Basic). Summarise an
article three ways (TextRank, BART/T5, LLM), compare two prompt techniques,
caption uploaded figures with BLIP, listen to a TTS audio summary, generate
a thumbnail, and flag unsupported sentences with an NLI faithfulness
guardrail.

Run locally:   python app.py        # or: gradio app.py
API keys (optional) go in the host's Secrets settings - never in this
file. See README.md.
"""

import json
import os
import tempfile

import gradio as gr

from core.u1_prompting import (LENGTH_PRESETS, PROMPT_TECHNIQUES,
                               STYLE_PRESETS, build_prompt,
                               latent_space_figure, split_sentences)
from core.u2_summarizers import (bart_summary, cross_attention_figure,
                                 llm_summary, textrank_summary)
from core.u3_multimodal import augment_article_with_captions, caption_images
from core.u4_media import generate_thumbnail, tts_summary
from core.u5_guardrail import check_faithfulness, hallucination_report

TITLE = "P22 - Abstractive vs Extractive Summariser"

SAMPLES = {}
try:
    with open(os.path.join(os.path.dirname(__file__),
                           "benchmark", "test_articles.json")) as _f:
        for _a in json.load(_f):
            SAMPLES[f"sample {_a['id']}: {_a['title']}"] = _a
except OSError:
    pass


def _sample_text(choice):
    a = SAMPLES.get(choice or "")
    return (a["article"], a["reference"]) if a else ("", "")


def _flags_markdown(name, verdicts):
    rep = hallucination_report(verdicts)
    lines = [f"**{name}** - {rep['sentences_flagged']} of "
             f"{rep['sentences_checked']} sentences could not be verified "
             f"against the source"]
    for v in verdicts:
        icon = "🚩" if v["flagged"] else "✅"
        lines.append(f"- {icon} {v['sentence']}  \n"
                     f"  entailment {v['entailment']}, "
                     f"contradiction {v['contradiction']}; "
                     f"closest source: _{v['best_source']}_")
    return "\n".join(lines)


def run_pipeline(choice, article, reference, figures, length, style,
                 technique, methods, extras, progress=gr.Progress()):
    article = (article or "").strip()
    if not article:
        raise gr.Error("Paste an article or pick a sample first.")
    work_article = article
    captions_md = ""

    # --- U3: BLIP figure captions injected into the article ---------------
    if figures:
        from PIL import Image
        progress(0.05, desc="Captioning figures with BLIP (U3)...")
        images = [Image.open(f.name if hasattr(f, "name") else f)
                  for f in figures]
        captions = caption_images(images)
        work_article = augment_article_with_captions(work_article, captions)
        captions_md = "### Figure captions (U3)\n" + "\n".join(
            f"- Figure {i + 1}: {c}" for i, c in enumerate(captions))

    # --- U2: three summarisation methods -----------------------------------
    summaries, backends = {}, {}
    steps = [m for m in ("TextRank", "BART/T5", "LLM") if m in methods]
    for i, m in enumerate(steps):
        progress(0.1 + 0.4 * i / max(len(steps), 1), desc=f"{m} summary...")
        if m == "TextRank":
            summaries["TextRank (extractive)"] = textrank_summary(
                work_article, sentence_count=5)
        elif m == "BART/T5":
            summaries["BART/T5 (abstractive)"] = bart_summary(
                work_article, length=length)
        elif m == "LLM":
            text, backend = llm_summary(
                build_prompt(work_article, technique, length, style))
            summaries["LLM"] = text
            backends["LLM"] = backend
    if not summaries:
        raise gr.Error("Select at least one method in the sidebar.")

    summaries_md = "### Summaries (U2)\n" + "\n\n".join(
        f"**{name}**"
        + (f"  \n_backend: {backends[name]}_" if name in backends else "")
        + f"\n\n{text}" for name, text in summaries.items())

    # --- U5: NLI faithfulness guardrail ------------------------------------
    progress(0.55, desc="NLI faithfulness check (U5)...")
    flags_md = ("### Faithfulness check (U5)\n"
                "_An NLI classifier signal: a flag means the sentence could "
                "not be verified against the source - not proof of error, "
                "and unflagged text is not guaranteed correct._\n\n")
    flags_md += "\n\n".join(
        _flags_markdown(name, check_faithfulness(work_article, text))
        for name, text in summaries.items())

    # --- ROUGE vs reference -------------------------------------------------
    rouge_md = ""
    if (reference or "").strip():
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                          use_stemmer=True)
        rouge_md = ("### ROUGE vs reference summary\n"
                    "| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |\n"
                    "|---|---|---|---|\n")
        for name, text in summaries.items():
            s = scorer.score(reference, text)
            rouge_md += (f"| {name} | {s['rouge1'].fmeasure:.3f} | "
                         f"{s['rouge2'].fmeasure:.3f} | "
                         f"{s['rougeL'].fmeasure:.3f} |\n")

    # --- U1: latent-space view + prompt-technique comparison ----------------
    latent_fig = None
    if "Latent-space view (U1)" in extras:
        progress(0.7, desc="Latent-space view (U1)...")
        try:
            latent_fig = latent_space_figure(
                split_sentences(work_article),
                {n: split_sentences(t) for n, t in summaries.items()})
        except Exception as e:
            latent_fig = None
            flags_md += f"\n\n_latent-space view unavailable: {e}_"

    prompts_md = ""
    if "Prompt-technique comparison (U1)" in extras and "LLM" in methods:
        progress(0.75, desc="Comparing prompt techniques (U1)...")
        prompts_md = "### Prompt-technique comparison (U1)\n"
        for key, (label, _) in PROMPT_TECHNIQUES.items():
            p = build_prompt(work_article, key, length, style)
            out, backend = llm_summary(p)
            prompts_md += (f"**{label}** (backend: {backend})\n\n{out}\n\n")

    # --- U2 deep layer: cross-attention heatmap ------------------------------
    attn_fig = None
    if "Attention heatmap (U2)" in extras and "BART/T5" in methods:
        progress(0.85, desc="Extracting cross-attention (U2)...")
        attn_fig = cross_attention_figure(work_article, length="short")

    # --- U4: TTS audio + generated thumbnail ---------------------------------
    audio_path, thumb_path, thumb_note = None, None, ""
    if "TTS audio + thumbnail (U4)" in extras and summaries:
        progress(0.92, desc="TTS audio + thumbnail (U4)...")
        first = next(iter(summaries.values()))
        tmp = tempfile.mkdtemp()
        audio_path = tts_summary(first, os.path.join(tmp, "summary.mp3"))
        if choice in SAMPLES:
            title = choice.split(": ", 1)[-1]
        else:
            title = split_sentences(article)[0][:80]
        thumb_path, backend = generate_thumbnail(
            title, first, os.path.join(tmp, "thumbnail.png"))
        thumb_note = f"thumbnail backend: {backend}"

    return (captions_md, summaries_md, flags_md, rouge_md, prompts_md,
            latent_fig, attn_fig, audio_path, thumb_path, thumb_note)


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}\nGenAI mini project (Roll 22 -> P22): summarise "
                "an article three ways and check which summaries stay "
                "faithful. Units U1-U5 included; see README/MODEL_CARD in "
                "the repo.")

    with gr.Row():
        with gr.Column(scale=2):
            choice = gr.Dropdown(["(paste your own)"] + list(SAMPLES),
                                 value="(paste your own)", label="Samples")
            article = gr.Textbox(label="Article text", lines=12)
            reference = gr.Textbox(
                label="Reference summary for ROUGE (optional)", lines=3)
            figures = gr.File(label="Figures to caption with BLIP (U3, "
                                    "optional)", file_count="multiple",
                              file_types=["image"])
        with gr.Column(scale=1):
            length = gr.Radio(list(LENGTH_PRESETS), value="medium",
                              label="Length (U1)")
            style = gr.Dropdown(list(STYLE_PRESETS), value="paragraph",
                                label="Style (U1)")
            technique = gr.Dropdown(
                list(PROMPT_TECHNIQUES), value="role_format",
                label="Prompt technique (U1)")
            methods = gr.CheckboxGroup(["TextRank", "BART/T5", "LLM"],
                                       value=["TextRank", "BART/T5", "LLM"],
                                       label="Methods (U2)")
            extras = gr.CheckboxGroup(
                ["Latent-space view (U1)", "Prompt-technique comparison (U1)",
                 "Attention heatmap (U2)", "TTS audio + thumbnail (U4)"],
                value=["Latent-space view (U1)"], label="Extras")
            run = gr.Button("Summarise", variant="primary")

    choice.change(_sample_text, choice, [article, reference])

    captions = gr.Markdown()
    summaries = gr.Markdown()
    flags = gr.Markdown()
    rouge = gr.Markdown()
    prompts = gr.Markdown()
    with gr.Row():
        latent = gr.Plot(label="Latent-space view (U1)")
        attn = gr.Plot(label="Cross-attention heatmap (U2)")
    with gr.Row():
        audio = gr.Audio(label="TTS audio summary (U4)", type="filepath")
        thumb = gr.Image(label="Generated thumbnail (U4)", type="filepath")
    thumb_note = gr.Markdown()

    run.click(run_pipeline,
              [choice, article, reference, figures, length, style,
               technique, methods, extras],
              [captions, summaries, flags, rouge, prompts,
               latent, attn, audio, thumb, thumb_note])

    gr.Markdown("Faithfulness flags come from an NLI classifier and mean "
                "'unverified against the source', not proven wrong. Optional "
                "API keys (GROQ_API_KEY, GEMINI_API_KEY, HF_TOKEN) are read "
                "from the host's secrets settings only.")

if __name__ == "__main__":
    demo.launch()
