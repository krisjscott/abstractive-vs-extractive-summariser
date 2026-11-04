"""Unit V - Responsible AI.

U5 task: "Flag unsupported sentences with an NLI model."
Each summary sentence is checked against the source article with a
natural-language-inference cross-encoder. A sentence is flagged as
unsupported (a potential hallucination) when no source sentence entails it.
This is the measured hallucination test plus the app's safety guardrail.
"""

from core.gpu_compat import GPU
import os

NLI_MODEL = os.environ.get("NLI_MODEL", "cross-encoder/nli-deberta-v3-xsmall")
ENTAILMENT_THRESHOLD = float(os.environ.get("NLI_THRESHOLD", "0.5"))

_nli = None
_label_ids = None


def _load_nli():
    global _nli, _label_ids
    if _nli is None:
        from sentence_transformers import CrossEncoder
        _nli = CrossEncoder(NLI_MODEL, max_length=512, device="cpu")
        id2label = {int(k): v.lower() for k, v in
                    _nli.model.config.id2label.items()}
        _label_ids = {name: i for i, name in id2label.items()}
    return _nli, _label_ids


def _max_overlap_sources(sentence: str, source_sentences: list, top_k: int = 3):
    """Pick the source sentences with the most word overlap (premises)."""
    words = set(sentence.lower().split())
    scored = sorted(
        source_sentences,
        key=lambda s: len(words & set(s.lower().split())), reverse=True)
    return scored[:top_k]


def check_faithfulness(source_text: str, summary_text: str,
                       sentence_splitter=None) -> list:
    """Return one verdict dict per summary sentence.

    Each verdict: {sentence, best_source, entailment, contradiction,
                   label, flagged}.
    """
    import numpy as np
    if sentence_splitter is None:
        from core.u1_prompting import split_sentences
        sentence_splitter = split_sentences
    model, label_ids = _load_nli()
    ent_id = label_ids.get("entailment", 1)
    con_id = label_ids.get("contradiction", 0)

    source_sents = [s for s in sentence_splitter(source_text) if len(s) > 10]
    summary_sents = [s for s in sentence_splitter(summary_text) if len(s) > 3]
    if not source_sents or not summary_sents:
        return []

    verdicts = []
    for sent in summary_sents:
        premises = _max_overlap_sources(sent, source_sents)
        pairs = [(p, sent) for p in premises]
        logits = np.asarray(model.predict(pairs))
        probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        best_i = int(probs[:, ent_id].argmax())
        entailment = float(probs[best_i, ent_id])
        contradiction = float(probs[best_i, con_id])
        flagged = entailment < ENTAILMENT_THRESHOLD
        verdicts.append({
            "sentence": sent,
            "best_source": premises[best_i],
            "entailment": round(entailment, 3),
            "contradiction": round(contradiction, 3),
            "label": "supported" if not flagged else "unsupported",
            "flagged": flagged,
        })
    return verdicts


def hallucination_report(verdicts: list) -> dict:
    """Aggregate the per-sentence verdicts into hallucination flags."""
    flagged = [v for v in verdicts if v["flagged"]]
    return {
        "sentences_checked": len(verdicts),
        "sentences_flagged": len(flagged),
        "flag_rate": round(len(flagged) / len(verdicts), 3) if verdicts else 0.0,
        "flagged_sentences": [v["sentence"] for v in flagged],
    }
