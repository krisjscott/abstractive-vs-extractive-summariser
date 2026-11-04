"""Unit I - Foundations of Generative AI.

Covers the assignment's U1 task ("Add length and style prompts") plus the
common Unit I layer rules:
  * at least 2 prompt techniques compared (PROMPT_TECHNIQUES below), and
  * one probability or latent-space view (latent_space_figure).
"""

import os

# ---------------------------------------------------------------------------
# Length and style presets (U1: "Add length and style prompts")
# ---------------------------------------------------------------------------

LENGTH_PRESETS = {
    "short": "about 2 sentences (max 60 words)",
    "medium": "about 5 sentences (max 150 words)",
    "detailed": "a detailed summary of about 10 sentences (max 300 words)",
}

STYLE_PRESETS = {
    "paragraph": "a single coherent paragraph in a neutral news tone",
    "bullet points": "a list of concise bullet points",
    "executive": "an executive briefing that leads with the key takeaway",
    "simple": "plain, simple English a high-school student could follow",
}

# Two prompt techniques that the app compares side by side (Unit I rule:
# "at least 2 prompt techniques compared").
PROMPT_TECHNIQUES = {
    "zero_shot": (
        "Zero-shot instruction",
        "Summarise the following article in {length} as {style}.\n\n"
        "Article:\n{article}\n\nSummary:",
    ),
    "role_format": (
        "Role + context + output-format prompt",
        "You are a careful news editor. Read the article below and write "
        "{length} as {style}.\n"
        "Rules: only use facts stated in the article, do not add numbers or "
        "names that are not in the text, and keep the original meaning.\n\n"
        "Article:\n{article}\n\nReturn ONLY the summary, no preamble.",
    ),
}


def build_prompt(article: str, technique: str = "role_format",
                 length: str = "medium", style: str = "paragraph") -> str:
    """Render the LLM summarisation prompt for one technique/length/style."""
    if technique not in PROMPT_TECHNIQUES:
        raise ValueError(f"unknown technique: {technique}")
    template = PROMPT_TECHNIQUES[technique][1]
    return template.format(
        length=LENGTH_PRESETS.get(length, LENGTH_PRESETS["medium"]),
        style=STYLE_PRESETS.get(style, STYLE_PRESETS["paragraph"]),
        article=article.strip(),
    )


def split_sentences(text: str):
    """Sentence splitter with an NLTK path and a regex fallback."""
    text = " ".join(text.split())
    try:
        import nltk
        try:
            return nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            return nltk.sent_tokenize(text)
    except Exception:
        import re
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
        return [p.strip() for p in parts if p.strip()]


def latent_space_figure(source_sentences, summary_sets, max_sentences=40):
    """Latent-space view (Unit I): sentence embeddings projected to 2D.

    source_sentences : list[str]  sentences of the original article
    summary_sets     : dict[str, list[str]]  method name -> summary sentences
    Returns a matplotlib Figure.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sentence_transformers import SentenceTransformer

    model_name = os.environ.get("EMBED_MODEL",
                                "sentence-transformers/all-MiniLM-L6-v2")
    model = SentenceTransformer(model_name)

    labels, texts = [], []
    for s in source_sentences[:max_sentences]:
        labels.append("article")
        texts.append(s)
    for method, sents in summary_sets.items():
        for s in sents[:10]:
            labels.append(method)
            texts.append(s)
    if len(texts) < 3:
        raise ValueError("need at least 3 sentences for a latent-space view")

    emb = np.asarray(model.encode(texts, normalize_embeddings=True))
    # 2D PCA via SVD (no sklearn needed at runtime)
    centered = emb - emb.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T

    fig, ax = plt.subplots(figsize=(7, 5))
    styles = {"article": ("o", "tab:gray")}
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    other = [l for l in dict.fromkeys(labels) if l != "article"]
    for i, m in enumerate(other):
        styles[m] = ("^", palette[i % len(palette)])
    for lab in dict.fromkeys(labels):
        idx = [i for i, l in enumerate(labels) if l == lab]
        marker, color = styles[lab]
        ax.scatter(coords[idx, 0], coords[idx, 1], marker=marker, c=color,
                   label=lab, alpha=0.8, edgecolors="white")
    ax.set_title("Latent-space view: article vs summary sentences "
                 "(MiniLM embeddings, PCA)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend()
    fig.tight_layout()
    return fig
