"""Unit II - GANs & LLMs (Track B deep layer: transformer internals).

Three summarisation methods compared (U2 task):
  1. TextRank          - classic extractive graph method (sumy)
  2. BART/T5           - encoder-decoder abstractive model (transformers)
  3. LLM               - instruction-tuned LLM via free API key (Groq/Gemini,
                         keys live in the host's secrets settings) with a
                         small local open model as the no-key fallback.

The common Unit II rule asks for one GAN/VAE element OR an attention
explanation; cross_attention_figure() provides the attention explanation
(it renders the encoder-decoder cross-attention that the abstractive model
used to produce the summary).
"""

import os

from core.gpu_compat import GPU

BART_MODEL = os.environ.get("BART_MODEL", "sshleifer/distilbart-cnn-6-6")
LLM_LOCAL_MODEL = os.environ.get("LLM_LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

LENGTH_TO_TOKENS = {"short": (60, 15), "medium": (150, 40), "detailed": (300, 80)}

_seq2seq = None
_seq2seq_tokenizer = None


def _load_seq2seq():
    global _seq2seq, _seq2seq_tokenizer
    if _seq2seq is None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        _seq2seq_tokenizer = AutoTokenizer.from_pretrained(BART_MODEL)
        dtype = {"float16": torch.float16, "float32": torch.float32}[
            os.environ.get("BART_DTYPE", "float32")]
        _seq2seq = AutoModelForSeq2SeqLM.from_pretrained(
            BART_MODEL, dtype=dtype,
            attn_implementation="eager")  # eager exposes cross-attention
        _seq2seq.eval()
    return _seq2seq, _seq2seq_tokenizer


# ---------------------------------------------------------------------------
# Method 1: TextRank (extractive)
# ---------------------------------------------------------------------------

def textrank_summary(text: str, sentence_count: int = 5) -> str:
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.summarizers.text_rank import TextRankSummarizer
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
    except LookupError:
        import nltk
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    sentences = summarizer(parser.document, sentence_count)
    return " ".join(str(s) for s in sentences)


# ---------------------------------------------------------------------------
# Method 2: BART/T5 encoder-decoder (abstractive)
# ---------------------------------------------------------------------------

def bart_summary(text: str, length: str = "medium") -> str:
    model, tokenizer = _load_seq2seq()
    max_new, min_new = LENGTH_TO_TOKENS.get(length, LENGTH_TO_TOKENS["medium"])
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=1024)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    import torch
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new,
                             min_new_tokens=min_new, num_beams=4,
                             no_repeat_ngram_size=3, early_stopping=True)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Method 3: LLM (API key from host secrets, local open model as fallback)
# ---------------------------------------------------------------------------

def llm_summary(prompt: str, max_tokens: int = 220):
    """Summarise with an instruction-tuned LLM.

    Returns (summary_text, backend_name). Backend order:
    Groq (GROQ_API_KEY) -> Gemini (GEMINI_API_KEY) -> local Qwen2.5 instruct.
    Keys are read from environment / Streamlit secrets only - never from
    the repository.
    """
    import requests

    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_key}"},
            json={"model": GROQ_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": max_tokens, "temperature": 0.3},
            timeout=60)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        return text, f"Groq ({GROQ_MODEL})"

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_MODEL}:generateContent?key={gemini_key}",
            json={"contents": [{"parts": [{"text": prompt}]}],
                  "generationConfig": {"maxOutputTokens": max_tokens,
                                       "temperature": 0.3}},
            timeout=60)
        r.raise_for_status()
        text = (r.json()["candidates"][0]["content"]["parts"][0]["text"]
                .strip())
        return text, f"Gemini ({GEMINI_MODEL})"

    return _local_llm_summary(prompt, max_tokens)


def _local_llm_summary(prompt: str, max_tokens: int = 220):
    # No API key configured: run a small open instruct model locally so the
    # app still works end to end with no key.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    global _causal, _causal_tokenizer
    try:
        _causal
    except NameError:
        _causal = None
        _causal_tokenizer = None
    if _causal is None:
        # fp16 keeps the fallback model small enough for free CPU hosts
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[os.environ.get("LLM_DTYPE",
                                                          "float32")]
        _causal_tokenizer = AutoTokenizer.from_pretrained(LLM_LOCAL_MODEL)
        _causal = AutoModelForCausalLM.from_pretrained(LLM_LOCAL_MODEL,
                                                       dtype=dtype)
        _causal.eval()
    messages = [{"role": "user", "content": prompt}]
    enc = _causal_tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
        return_dict=True)
    device = next(_causal.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = _causal.generate(**enc, max_new_tokens=max_tokens,
                               do_sample=False)
    text = _causal_tokenizer.decode(
        out[0][enc["input_ids"].shape[-1]:],
        skip_special_tokens=True).strip()
    return text, f"local {LLM_LOCAL_MODEL}"


# ---------------------------------------------------------------------------
# Unit II deep layer (Track B): cross-attention explanation
# ---------------------------------------------------------------------------

def cross_attention_figure(text: str, length: str = "short",
                           max_src_tokens: int = 60):
    """Render averaged last-layer decoder cross-attention as a heatmap.

    The rows are generated summary tokens, the columns are source tokens;
    brighter cells mean the decoder focused on that source token while
    writing the summary token.
    """
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model, tokenizer = _load_seq2seq()
    max_new, _ = LENGTH_TO_TOKENS.get(length, LENGTH_TO_TOKENS["short"])
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    src_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new,
                             output_attentions=True,
                             return_dict_in_generate=True)
    tgt_tokens = tokenizer.convert_ids_to_tokens(gen.sequences[0][1:])
    # cross_attentions: one tuple per generated step -> (layers, B, H, 1, S)
    mats = []
    for step in gen.cross_attentions:
        last_layer = step[-1][0]          # (heads, 1, src)
        mats.append(last_layer.mean(dim=0)[0].numpy())   # (src,)
    attn = np.stack(mats)                 # (tgt, src)
    src_tokens = src_tokens[:max_src_tokens]
    attn = attn[:, :max_src_tokens]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(attn, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=90, fontsize=6)
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels(tgt_tokens, fontsize=6)
    ax.set_title(f"Cross-attention of {BART_MODEL} (last layer, head average)")
    ax.set_xlabel("source tokens")
    ax.set_ylabel("generated summary tokens")
    fig.tight_layout()
    return fig
