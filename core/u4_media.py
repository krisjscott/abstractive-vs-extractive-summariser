"""Unit IV - Generative AI tools & techniques.

U4 task: "Produce a TTS audio summary and generated thumbnail."
  * tts_summary         - gTTS speech synthesis of the summary (mp3)
  * generate_thumbnail  - SD-Turbo text-to-image via the Hugging Face
                          Inference API (HF_TOKEN in the host's secrets),
                          with a clearly-labelled PIL title card as the
                          offline fallback so the app never breaks.
"""

from core.gpu_compat import GPU
import os

SD_MODEL = os.environ.get("SD_MODEL", "stabilityai/sd-turbo")


def tts_summary(text: str, out_path: str = "summary_audio.mp3") -> str:
    """Synthesise the summary as speech; returns the mp3 path."""
    from gtts import gTTS
    gTTS(text=text, lang="en").save(out_path)
    return out_path


def _hf_inference_thumbnail(prompt: str, out_path: str) -> bool:
    """Try the HF Inference API (free tier, token from secrets)."""
    import requests
    token = os.environ.get("HF_TOKEN")
    if not token:
        return False
    r = requests.post(
        f"https://api-inference.huggingface.co/models/{SD_MODEL}",
        headers={"Authorization": f"Bearer {token}"},
        json={"inputs": prompt, "parameters": {"num_inference_steps": 2}},
        timeout=120)
    if r.status_code != 200 or "image" not in r.headers.get(
            "content-type", ""):
        return False
    with open(out_path, "wb") as f:
        f.write(r.content)
    return True


def _local_diffusers_thumbnail(prompt: str, out_path: str,
                               force_cpu: bool = False) -> bool:
    """Local SD-Turbo generation (real diffusion, no API). Runs on GPU when
    one is attached, otherwise a genuine CPU pass (slow: 1-3 minutes)."""
    if os.environ.get("LOCAL_DIFFUSERS", "1") == "0":
        return False
    import gc
    pipe = None
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
        on_gpu = (not force_cpu) and torch.cuda.is_available()
        pipe = AutoPipelineForText2Image.from_pretrained(
            SD_MODEL, torch_dtype=torch.float16 if on_gpu else torch.float32)
        if on_gpu:
            pipe = pipe.to("cuda")
        image = pipe(prompt=prompt, num_inference_steps=2,
                     guidance_scale=0.0).images[0]
        image.save(out_path)
        return True
    except Exception:
        return False
    finally:
        del pipe
        gc.collect()


@GPU
def _gpu_diffusers_thumbnail(prompt: str, out_path: str) -> bool:
    """ZeroGPU entry point: same real SD-Turbo generation on the attached
    GPU. Raises when the host cannot attach a GPU (quota/capacity), which
    generate_thumbnail catches before the genuine CPU fallback."""
    return _local_diffusers_thumbnail(prompt, out_path)


def _title_card_thumbnail(title: str, out_path: str) -> None:
    """Offline fallback: a generated gradient title card (PIL only)."""
    from PIL import Image, ImageDraw, ImageFont
    import textwrap

    w, h = 768, 432
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    top, bottom = (24, 32, 74), (9, 110, 120)
    for y in range(h):  # vertical gradient
        t = y / h
        draw.line([(0, y), (w, y)], fill=tuple(
            int(a + (b - a) * t) for a, b in zip(top, bottom)))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 42)
        small = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
        small = font
    draw.multiline_text((40, 120), "\n".join(textwrap.wrap(title, 34)),
                        font=font, fill="white", spacing=8)
    draw.text((40, h - 56), "Generated thumbnail (title-card fallback)",
              font=small, fill=(200, 210, 225))
    img.save(out_path)


def generate_thumbnail(title: str, summary: str,
                       out_path: str = "thumbnail.png"):
    """Generate a thumbnail; returns (path, backend_used)."""
    prompt = (f"editorial illustration for a news article titled "
              f"'{title}'. Flat modern style, no text. Theme: "
              f"{summary[:200]}")
    if _hf_inference_thumbnail(prompt, out_path):
        return out_path, f"HF Inference API ({SD_MODEL})"
    try:
        if _gpu_diffusers_thumbnail(prompt, out_path):
            return out_path, f"local diffusers on GPU ({SD_MODEL})"
    except Exception:
        pass  # no GPU quota/capacity on this host - try a real CPU pass
    if _local_diffusers_thumbnail(prompt, out_path, force_cpu=True):
        return out_path, f"local diffusers on CPU ({SD_MODEL}, slow)"
    _title_card_thumbnail(title, out_path)
    return out_path, "PIL title-card fallback (diffusion unavailable)"
