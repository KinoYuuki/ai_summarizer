import re, os, requests, sys, platform
from bs4 import BeautifulSoup
from transformers import pipeline, logging
from summarizer.hardware.detector import detect_hardware
from summarizer.hardware.optimizers import apply_amd_optimizations, apply_cpu_optimizations

# ======================
# AMD PERFORMANCE CONFIG
# ======================
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
logging.set_verbosity_error()

def clean_summary(text):
    """Enhanced text cleaning that preserves version numbers and technical terms"""
    # Step 1: Protect version numbers and technical terms with dots
    protected = []

    def protect_version(match):
        protected.append(match.group(0))
        return f"@@PROTECTED_{len(protected) - 1}@@"

    # Patterns to protect (version numbers, abbreviations, etc.)
    version_pattern = r'\b(?:[a-zA-Z]+ \d+\.\d+(?:\.\d+)*|\d+\.\d+\.\d+)\b'
    text = re.sub(version_pattern, protect_version, text)

    # Step 2: Fix general punctuation spacing
    text = re.sub(r'\s+([.,!?])', r'\1', text)

    # Step 3: Capitalize sentences properly
    sentences = []
    for s in text.split('.'):
        s = s.strip()
        if s:
            # Handle case where sentence starts with protected token
            if s.startswith('@@PROTECTED_'):
                sentences.append(s)
            else:
                sentences.append(s[0].upper() + s[1:])

    text = '. '.join(sentences).strip()

    # Step 4: Restore protected terms
    for i, term in enumerate(protected):
        text = text.replace(f'@@PROTECTED_{i}@@', term)

    # Ensure proper final punctuation
    if text and not text[-1] in '.!?':
        text += '.'

    return text

def format_bullets(text):
    """Convert sentences to bullet points reliably"""
    import re
    # Split at .!? but avoid splitting at decimals/abbreviations
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    return "• " + "\n• ".join(s.strip() for s in sentences if s.strip())

def generate_summary(args):
    try:
        # ===== START DEBUG CODE =====
        if args.debug:
            print("\n🔍 DEBUG MODE")
            print("=" * 50)
            print(f"Python: {sys.version}")
            print(f"Platform: {platform.platform()}")
            print(f"CPU Cores: {os.cpu_count()}")

        hw_config = detect_hardware()
        if args.debug:  # Show hardware detection results
            print(f"\n⚙️ Hardware Config: {hw_config}")
            print(f"🔧 Using device: {hw_config['device'].upper()}")
            print("=" * 50 + "\n")
        # ===== END DEBUG CODE =====

        # Apply hardware-specific optimizations
        if hw_config.get("amd_optimized"):
            apply_amd_optimizations()
        elif hw_config["device"] == "cpu":  # NEW: CPU optimization branch
            apply_cpu_optimizations()

        summarizer = pipeline(
            task="summarization",
            model="sshleifer/distilbart-cnn-6-6",
            device=hw_config["device"],
            framework=hw_config["framework"],
            torch_dtype=hw_config.get("torch_dtype", None),
        )

        # Memory management
        summarizer.tokenizer.model_max_length = 512
        summarizer.model.config.max_length = 200

        # Get input text
        if args.text:
            if os.path.exists(args.text):
                with open(args.text, 'r', encoding='utf-8') as f:
                    input_text = f.read()
                print(f"📖 Read {len(input_text.split())} words from file")
            else:
                input_text = args.text
        else:
            response = requests.get(args.url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
            soup = BeautifulSoup(response.text, 'html.parser')
            input_text = ' '.join(p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip())

        # Length handling
        word_count = len(input_text.split())
        if word_count < 30:
            print("\n[Note] Input very short - returning cleaned text")
            summary = clean_summary(input_text)
        else:
            max_len = min(150, max(40, int(word_count * 0.4)))
            min_len = min(30, max(15, int(max_len * 0.6)))

            summary = summarizer(
                input_text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False
            )[0]['summary_text']
            summary = clean_summary(summary)

            # Quality fallback if summary is too short
            if len(summary.split()) < 10 and word_count > 30:
                summary = summarizer(
                    input_text,
                    max_length=100,
                    min_length=40,
                    do_sample=True
                )[0]['summary_text']
                summary = clean_summary(summary)

        summary_text = format_bullets(summary) if args.bullets else summary

        # Final output
        print("\n" + "=" * 50)
        print(f"📝 SUMMARY ({len(summary.split())} words)".center(50))
        print("=" * 50)

        if args.bullets:
            # Simple bullet conversion (works for most cases)
            for sentence in summary.split('.'):
                if sentence.strip():  # Skip empty strings
                    print(f"• {sentence.strip()}.")
        else:
            print(f"\n{summary}\n")  # Original format

        print("=" * 50)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"\n💾 Saved to: {os.path.abspath(args.output)}")

    except Exception as e:
        print(f"\n🔥 Error: {str(e)}")
        sys.exit(1)