import os, torch

def detect_hardware():
    """Auto-detects available hardware and returns optimal settings"""
    config = {
        "device": "cpu",
        "threads": os.cpu_count(),
        "framework": "pt"
    }

    if torch.cuda.is_available():
        config.update({
            "device": "cuda",
            "torch_dtype": "auto"
        })
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        config.update({
            "device": "mps"  # Apple Silicon
        })

    # AMD ROCm detection (optional)
    if torch.version.hip:
        config.update({
            "device": "cuda",
            "amd_optimized": True
        })

    return config