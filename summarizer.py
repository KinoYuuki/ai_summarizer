import sys
from summarizer.engine import generate_summary
from summarizer.cli import create_parser

def main():
    parser = create_parser()
    args = parser.parse_args()

    if not args.text and not args.url:
        parser.print_help()
        sys.exit(1)

    generate_summary(args)

if __name__ == "__main__":
    main()