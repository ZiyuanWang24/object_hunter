from __future__ import annotations

import numpy as np


class DINOv2Embedder:
    def __init__(self, model_name="facebook/dinov2-small", device="auto"):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as e:
            raise RuntimeError("DINOv2 backend requires torch, transformers and Pillow.") from e
        self.torch = torch
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        except Exception as e:
            raise RuntimeError(
                f"Could not load DINOv2 weights '{model_name}'. Cache the Hugging Face model "
                "locally or allow access to huggingface.co, then retry."
            ) from e

    def embed(self, crops):
        if not crops: return []
        inputs = self.processor(images=crops, return_tensors="pt").to(self.device)
        if self.device.startswith("cuda"):
            context = self.torch.autocast(device_type="cuda", dtype=self.torch.float16)
        else:
            context = self.torch.autocast(device_type="cpu", enabled=False)
        with self.torch.inference_mode(), context:
            output = self.model(**inputs).last_hidden_state[:, 0]
            output = self.torch.nn.functional.normalize(output, dim=-1)
        return [v.detach().float().cpu().numpy() for v in output]


class NullEmbedder:
    def embed(self, crops): return [None for _ in crops]
