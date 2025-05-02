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
    input_group.add_argument("--inspect-training", action="store_true",
                             help="Validate training data structure and show statistics")
    input_group.add_argument("--train", action="store_true",
                             help="Train the summarization model")

    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--output", help="Save summary to file")
    output_group.add_argument("--bullets", action="store_true",
                              help="Format as bullet points")
    output_group.add_argument("--length", choices=['short', 'medium', 'long'],
                              default='medium', help="Summary length preference")

    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument("--debug", action="store_true",
                              help="Show detailed processing info")
    config_group.add_argument('--model',
                              choices=['fast', 'quality', 'multilingual'],
                              default='fast',
                              help="Model: 'fast'(default)|'quality'|'multilingual'")

    return parser