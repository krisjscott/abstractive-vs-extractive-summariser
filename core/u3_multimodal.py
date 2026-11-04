"""Unit III - GPT, BERT & multimodal models.

U3 task: "Caption figures with BLIP so they enter the summary."
Figures uploaded with an article are captioned by BLIP
(Salesforce/blip-image-captioning-base, a vision-encoder + text-decoder
model) and the captions are injected into the article context before
summarisation, so visual information reaches the summary.
"""

import os

from core.gpu_compat import GPU

BLIP_MODEL = os.environ.get("BLIP_MODEL", "Salesforce/blip-image-captioning-base")

_blip = None


def _load_blip():
    global _blip
    if _blip is None:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor
        dtype = {"float16": torch.float16, "float32": torch.float32}[
            os.environ.get("BLIP_DTYPE", "float32")]
        _blip = {
            "processor": BlipProcessor.from_pretrained(BLIP_MODEL),
            "model": BlipForConditionalGeneration.from_pretrained(
                BLIP_MODEL, dtype=dtype),
        }
        _blip["model"].eval()
    return _blip


def caption_image(pil_image) -> str:
    """Caption one PIL image with BLIP."""
    import torch
    pack = _load_blip()
    inputs = pack["processor"](images=pil_image.convert("RGB"),
                               return_tensors="pt")
    device = next(pack["model"].parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = pack["model"].generate(**inputs, max_new_tokens=40)
    return pack["processor"].decode(out[0], skip_special_tokens=True).strip()


def caption_images(pil_images) -> list:
    """Caption a list of PIL images; returns one caption per image."""
    return [caption_image(img) for img in pil_images]


def augment_article_with_captions(article: str, captions: list) -> str:
    """Inject figure captions into the article so they enter the summary."""
    captions = [c for c in captions if c]
    if not captions:
        return article
    lines = "\n".join(f"Figure {i + 1}: {c}." for i, c in enumerate(captions))
    return (article.strip()
            + "\n\nThe article contains the following figures:\n" + lines)
