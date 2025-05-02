import argparse

def create_parser():
    parser = argparse.ArgumentParser(description="Text Summarizer")
    parser.add_argument("--text", help="Input text")
    parser.add_argument("--url", help="URL to summarize")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--debug", action="store_true", help="Show hardware details")
    parser.add_argument("--bullets", action="store_true", help="Format output as bullet points")
    return parser