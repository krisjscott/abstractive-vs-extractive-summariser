"""ZeroGPU compatibility shim (deployment hardware).

On Hugging Face ZeroGPU hardware the real `spaces` package moves each
decorated function onto a dynamically attached Nvidia GPU for the duration
of the call. Off ZeroGPU (local CPU, CI) `spaces` is a no-op passthrough;
if it is not installed at all we fall back to an identity decorator, so the
same code runs unchanged on a laptop and on the Space.
"""

try:
    import spaces
    GPU = spaces.GPU
except Exception:  # spaces not installed
    def GPU(fn=None, **_kwargs):
        if fn is None:
            return lambda f: f
        return fn
