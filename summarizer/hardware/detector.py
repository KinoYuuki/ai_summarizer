import os
import torch
from functools import lru_cache

@lru_cache(maxsize=1)
def detect_hardware():
    """Cached hardware detection with comprehensive device support"""
    config = {
        "device": "cpu",
        "threads": max(1, os.cpu_count() - 1),  # Leave 1 core free
        "framework": "pt"
    }

    if torch.cuda.is_available():
        config.update({
            "device": "cuda",
            "torch_dtype": "auto",
            "gpu_count": torch.cuda.device_count()
        })
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        config["device"] = "mps"

    # AMD ROCm detection
    if torch.version.hip:
        config.update({
            "device": "cuda",
            "amd_optimized": True,
            "rocm_version": torch.version.hip
        })

    return config