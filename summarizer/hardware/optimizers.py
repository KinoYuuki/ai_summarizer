import os
import platform
from typing import Dict

def apply_amd_optimizations(config: Dict):
    os.environ.update({
        'OMP_NUM_THREADS': str(config['threads']),
        'TOKENIZERS_PARALLELISM': 'false'
    })

def apply_nvidia_optimizations(config: Dict):
    """NVIDIA-specific settings"""
    os.environ.update({
        'CUDA_LAUNCH_BLOCKING': '1',
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true'
    })

def apply_cpu_optimizations(config: Dict):
    os.environ.update({
        'OMP_NUM_THREADS': str(config['threads']),
        'NUMEXPR_NUM_THREADS': str(config['threads']),
        'TF_NUM_INTEROP_THREADS': '1',
        'TF_NUM_INTRAOP_THREADS': str(config['threads'])
    })