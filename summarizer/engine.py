import re, sys, requests, time
from pathlib import Path
from typing import Dict
from bs4 import BeautifulSoup
from transformers import pipeline
from summarizer.hardware.detector import detect_hardware
from summarizer.hardware.optimizers import apply_amd_optimizations, apply_cpu_optimizations
from summarizer.constants import MODELS, VERSION_PATTERN, PUNCTUATION_PATTERN

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
    print("\n" + "=" * 50)
    print(f"📝 SUMMARY ({len(summary.split())} words)".center(50))
    print("=" * 50)

    if args.bullets:
        print(format_bullets(summary))
    else:
        print(clean_summary(summary))

    print("=" * 50)


def get_input_text(args) -> str:
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


def initialize_model(config: Dict, debug: bool = False, model_key: str = 'fast'):
    """Initialize the summarization pipeline with model selection"""
    model_params = {
        'task': 'summarization',
        'model': MODELS[model_key],
        'device': config['device'],
        'framework': config['framework'],
        'torch_dtype': config.get('torch_dtype')
    }

    if debug:
        print(f"\n⚙️ Model Config: {model_params}")
        print(f"Selected Model: {model_key} -> {MODELS[model_key]}")

    summarizer = pipeline(**model_params)
    summarizer.tokenizer.model_max_length = 512
    summarizer.model.config.max_length = 200
    return summarizer


def generate_summary(args, model_key='fast'):
    summarizer = None
    try:
        # Start timer right before heavy processing
        if args.debug:
            start_time = time.time()

        # Process input first to check length
        input_text = get_input_text(args)
        word_count = len(input_text.split())

        if model_key not in MODELS:
            raise ValueError(f"Invalid model key: {model_key}. Choose from {list(MODELS.keys())}")

        # Handle short texts (original behavior)
        if word_count < 25:  # Adjust threshold as needed
            if args.debug:  # Add debug info for short text
                print(f"\nℹ️ Debug: Text too short ({word_count} words < 25)")
                print(f"Selected model: {model_key} ({MODELS[model_key]})")
            print("\nℹ️  Input is too short - returning original text:")
            print("=" * 50)
            print(input_text)
            print("=" * 50)
            return

        # Only proceed with hardware setup if text is long enough
        hw_config = detect_hardware()

        # Apply optimizations
        if hw_config.get("amd_optimized"):
            apply_amd_optimizations(hw_config)
        elif hw_config["device"] == "cpu":
            apply_cpu_optimizations(hw_config)

        # Initialize pipeline
        summarizer = initialize_model(hw_config, args.debug, model_key)

        length_params = {
            'short': 0.3,  # 30% of original
            'medium': 0.4,  # 40% of original
            'long': 0.6  # 60% of original
        }

        compression = length_params[args.length]

        summary = summarizer(
            input_text,
            max_length=min(200, max(40, int(word_count * compression))),
            min_length=min(30, max(15, int(word_count * compression * 0.5))),
            do_sample=False
        )[0]['summary_text']

        if args.debug:
            print(f"\n⚙️ Hardware Config: {hw_config}")
            print(f"⚙️ Selected Model: {model_key} -> {MODELS[model_key]}")
            elapsed = time.time() - start_time
            print(f"⚡ Processing time: {elapsed:.2f}s | Speed: {len(input_text)/elapsed:.0f} chars/sec")

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