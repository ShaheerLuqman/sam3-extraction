"""Sanity check: torch, CUDA, and SAM 3 checkpoint loading."""
import sys
import torch

print(f"python        : {sys.version.split()[0]}")
print(f"torch         : {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device        : {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"vram          : {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")

try:
    import sam3
    from sam3.model_builder import build_sam3_image_model  # noqa: F401
    print("sam3 (native) : importable")
except Exception as e:
    print(f"sam3 (native) : NOT IMPORTABLE ({type(e).__name__}: {e})")

for mod in ("cv2", "matplotlib", "gradio"):
    try:
        __import__(mod)
        print(f"{mod:<14}: ok")
    except ImportError:
        print(f"{mod:<14}: NOT INSTALLED")
