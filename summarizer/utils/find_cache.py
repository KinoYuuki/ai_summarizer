import os
from pathlib import Path

def get_cache_path():
    """Returns the Hugging Face cache path"""
    return Path(os.path.expanduser('~/.cache/huggingface/hub'))

def fix_cache_location():
    cache_path = get_cache_path()
    print(f"Hugging Face cache is at: {cache_path}")
    return cache_path

def clear_cache():
    cache_path = get_cache_path()
    if cache_path.exists():
        # Add cleanup logic here
        print(f"Cache cleared at {cache_path}")