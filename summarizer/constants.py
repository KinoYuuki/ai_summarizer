__version__ = "1.1.0"
VERSION = __version__

import re

MODELS = {
    'fast': 'sshleifer/distilbart-cnn-6-6',  # Default (current)
    'quality': 'facebook/bart-large-cnn',     # Better accuracy
    'multilingual': 'mbart-large-50'         # Non-English support
}

VERSION_PATTERN = re.compile(r'\b(?:[a-zA-Z]+ \d+\.\d+(?:\.\d+)*|\d+\.\d+\.\d+)\b')
PUNCTUATION_PATTERN = re.compile(r'\s+([.,!?])')