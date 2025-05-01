import os, platform

def apply_amd_optimizations():
    """AMD-specific performance tweaks"""
    os.environ.update({
        'OMP_NUM_THREADS': str(os.cpu_count()),
        'TOKENIZERS_PARALLELISM': 'false',
        'HF_HUB_DISABLE_SYMLINKS_WARNING': '1'
    })

def apply_nvidia_optimizations():
    """NVIDIA-specific performance tweaks"""
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better debug if needed

def apply_cpu_optimizations():
    """Optimizations for CPU-only systems"""
    os.environ.update({
        'OMP_NUM_THREADS': str(os.cpu_count()),  # Use all cores
        'TOKENIZERS_PARALLELISM': 'false',       # Prevent tokenizer conflicts
        'NUMEXPR_NUM_THREADS': str(min(4, os.cpu_count())),  # Limit NumPy threads
        'TF_NUM_INTEROP_THREADS': '1',           # Better TensorFlow interop
        'TF_NUM_INTRAOP_THREADS': str(os.cpu_count())  # TensorFlow thread count
    })
    if platform.system() == 'Linux':
        os.environ['OMP_SCHEDULE'] = 'STATIC'    # Better Linux thread scheduling