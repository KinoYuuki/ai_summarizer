import argparse

def create_parser():
    parser = argparse.ArgumentParser(
        description="AI Text Summarizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Direct text input")
    input_group.add_argument("--file", type=str, help="Path to .txt file")
    input_group.add_argument("--url", help="URL to summarize")
    input_group.add_argument("--train-data", action="store_true",
                             help="Inspect training data structure")

    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--output", help="Save summary to file")
    output_group.add_argument("--bullets", action="store_true",
                              help="Format as bullet points")

    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument("--debug", action="store_true",
                              help="Show detailed processing info")

    return parser