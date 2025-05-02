import sys
from summarizer.engine import generate_summary
from summarizer.cli import create_parser
from summarizer.training import train_model, inspect_training_data

def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.inspect_training:
        inspect_training_data(debug=args.debug)
        return
    elif args.train:
        train_model(debug=args.debug)

    if args.debug:
        print("\n🔍 DEBUG MODE ACTIVATED")
        from summarizer.hardware.detector import detect_hardware
        hw_config = detect_hardware()
        print(f"⚙️ Hardware: {hw_config['device'].upper()} | Threads: {hw_config['threads']}")
        print("="*50)

    if not (args.text or args.url or args.file):
        parser.print_help()
        sys.exit(1)

    try:
        generate_summary(args)
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        sys.exit(130)

if __name__ == "__main__":
    main()