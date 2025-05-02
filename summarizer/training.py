from transformers import T5ForConditionalGeneration, T5Tokenizer

def inspect_training_data(debug=False):
    try:
        from datasets import load_dataset
        import os
        from tqdm import tqdm

        print("\n🔍 TRAINING DATA VALIDATION")

        # 1. Verify files
        required_files = {
            "inputs": "training_data/inputs.txt",
            "summaries": "training_data/summaries.txt"
        }

        for name, path in required_files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {name} file at {path}")
            file_size = os.path.getsize(path) / 1024
            print(f"✓ Found {name}: {file_size:.1f} KB")
            if file_size < 0.5:  # Warn if suspiciously small
                print(f"  ⚠️ Warning: {name} seems very small - check for content")

        # 2. Load and clean data
        def load_and_clean(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]

        inputs = load_and_clean(required_files["inputs"])
        summaries = load_and_clean(required_files["summaries"])

        # 3. Validate pairs
        if len(inputs) != len(summaries):
            raise ValueError(
                f"Mismatched pairs: {len(inputs)} inputs vs {len(summaries)} summaries\n"
                "Each input must have exactly one summary line"
            )

        # 4. Calculate stats
        stats = {
            "valid_pairs": len(inputs),
            "input_lengths": [len(x) for x in inputs],
            "summary_lengths": [len(x) for x in summaries],
            "empty_removed": None
        }

        print("\n📊 Cleaned Data Stats:")
        print(f"- Valid text pairs: {stats['valid_pairs']}")
        print(f"- Avg input length: {sum(stats['input_lengths']) / max(1, len(stats['input_lengths'])):.0f} chars")
        print(
            f"- Avg summary length: {sum(stats['summary_lengths']) / max(1, len(stats['summary_lengths'])):.0f} chars")
        print(f"- Compression ratio: {sum(stats['summary_lengths']) / sum(stats['input_lengths']):.1%}")

        # 5. Show samples
        print("\n🔬 Sample Pairs:")
        for i in range(min(3, len(inputs))):
            print(f"\nPair {i + 1}:")
            print(f"Input: {inputs[i][:100]}{'...' if len(inputs[i]) > 100 else ''}")
            print(f"Summary: {summaries[i]}")

        return {"train": inputs, "validation": summaries}

    except Exception as e:
        print(f"\n❌ Validation Failed: {str(e)}")
        print("\n💡 Fix your data files:")
        print("1. Remove empty lines")
        print("2. Ensure equal line counts")
        print("3. Example valid structure:")
        print("inputs.txt:")
        print("The quick brown fox...\nPython is...\n[text3]")
        print("summaries.txt:")
        print("Fox jumps\nPython lang\n[summary3]")
        return None

def train_model(debug=False):
    """Actual training implementation"""
    print("\n🚀 Starting Training" + (" (debug)" if debug else ""))

    # 1. Load data
    data = inspect_training_data(debug)
    if not data:
        return

    # 2. Initialize model
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    # 3. Training loop
    for epoch in range(3):
        print(f"Epoch {epoch + 1}")
        for input_text, target_text in zip(data["train"], data["validation"]):
            inputs = tokenizer("summarize: " + input_text, return_tensors="pt", truncation=True)
            labels = tokenizer(target_text, return_tensors="pt", truncation=True)

            outputs = model(**inputs, labels=labels["input_ids"])
            print(f"Loss: {outputs.loss.item():.4f}", end="\r")

    # 4. Save
    model.save_pretrained("models/")
    print("\n✅ Training complete")