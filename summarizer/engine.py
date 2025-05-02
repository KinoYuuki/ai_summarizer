import re, sys, requests
from pathlib import Path
from typing import Dict
from bs4 import BeautifulSoup
from transformers import pipeline
from summarizer.hardware.detector import detect_hardware
from summarizer.hardware.optimizers import apply_amd_optimizations, apply_cpu_optimizations

# Constants
DEFAULT_MODEL = "sshleifer/distilbart-cnn-6-6"
VERSION_PATTERN = re.compile(r'\b(?:[a-zA-Z]+ \d+\.\d+(?:\.\d+)*|\d+\.\d+\.\d+)\b')
PUNCTUATION_PATTERN = re.compile(r'\s+([.,!?])')


def clean_summary(text: str) -> str:
    """Enhanced text cleaning with version number preservation"""
    protected = []

    def protect(match):
        protected.append(match.group(0))
        return f"@@PROTECTED_{len(protected) - 1}@@"

    text = VERSION_PATTERN.sub(protect, text)
    text = PUNCTUATION_PATTERN.sub(r'\1', text)

    sentences = []
    for s in text.split('.'):
        s = s.strip()
        if s:
            sentences.append(s[0].upper() + s[1:] if not s.startswith('@@PROTECTED_') else s)

    text = '. '.join(sentences)
    for i, term in enumerate(protected):
        text = text.replace(f'@@PROTECTED_{i}@@', term)

    return text + ('' if text.endswith(('.', '!', '?')) else '.')


def format_bullets(text: str) -> str:
    """Convert text to bullet points"""
    sentences = [s.strip() for s in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text) if s.strip()]
    return '\n'.join(f"• {s}." for s in sentences)


def print_output(summary: str, args) -> None:
    """Print formatted output based on arguments"""
    print("\n" + "=" * 50)
    print(f"📝 SUMMARY ({len(summary.split())} words)".center(50))
    print("=" * 50)

    if args.bullets:
        print(format_bullets(summary))
    else:
        print(clean_summary(summary))

    print("=" * 50)


def get_input_text(args) -> str:
    """Get input text from file, direct input, or URL"""
    if args.file:
        path = Path(args.file)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path.read_text(encoding='utf-8')
    elif args.text:
        return args.text
    else:
        response = requests.get(args.url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return ' '.join(p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip())


def initialize_model(config: Dict, debug: bool = False):
    """Initialize the summarization pipeline"""
    model_params = {
        'task': 'summarization',
        'model': DEFAULT_MODEL,
        'device': config['device'],
        'framework': config['framework'],
        'torch_dtype': config.get('torch_dtype')
    }

    if debug:
        print(f"\n⚙️ Model Config: {model_params}")

    summarizer = pipeline(**model_params)
    summarizer.tokenizer.model_max_length = 512
    summarizer.model.config.max_length = 200
    return summarizer

def generate_summary(args):
    """Main summary generation function with resource cleanup"""
    summarizer = None
    try:
        # Hardware setup
        hw_config = detect_hardware()
        if args.debug:
            print(f"\n⚙️ Hardware Config: {hw_config}")

        # Apply optimizations
        if hw_config.get("amd_optimized"):
            apply_amd_optimizations(hw_config)
        elif hw_config["device"] == "cpu":
            apply_cpu_optimizations(hw_config)

        # Initialize pipeline
        summarizer = initialize_model(hw_config, args.debug)

        # Process input
        input_text = get_input_text(args)
        if not input_text.strip():
            raise ValueError("No meaningful text extracted from input")

        # Generate summary
        summary = summarizer(
            input_text,
            max_length=min(150, max(40, int(len(input_text.split()) * 0.4))),
            min_length=min(30, max(15, int(len(input_text.split()) * 0.2))),
            do_sample=False
        )[0]['summary_text']

        # Format and print output
        print_output(summary, args)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        if summarizer is not None:
            del summarizer
            if args.debug:
                print("\n🧹 Cleaned up model resources")