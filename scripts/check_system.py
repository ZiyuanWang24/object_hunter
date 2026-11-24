from __future__ import annotations
import importlib.util, platform, sys
print(f"Python: {sys.version.split()[0]} ({sys.executable})")
print(f"OS: {platform.platform()}")
for name in ["numpy","cv2","scipy","yaml","torch","ultralytics","transformers"]: print(f"{name}: {'installed' if importlib.util.find_spec(name) else 'missing'}")
try:
 import torch; print(f"CUDA available: {torch.cuda.is_available()}; torch CUDA: {torch.version.cuda}; GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
except ImportError: pass
