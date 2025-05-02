# AI Summarizer 🔍→📝

[![Version](https://img.shields.io/badge/Version-1.1.0-blue)](https://github.com/KinoYuuki/ai_summarizer)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/Framework-Transformers-orange)](https://huggingface.co/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered tool that generates concise summaries from long articles or documents using NLP techniques.

## Features ✨
- **Extractive summarization** (keeps key sentences)
- Supports both plain text and URL input
- Adjustable summary length (short/medium/long)
- Lightweight & fast processing

## Get Started
1. **Installation**:
   ```bash
   git clone https://github.com/KinoYuuki/ai_summarizer.git
   cd ai_summarizer
   pip install -r requirements.txt
   ```
2. **Command Flags 🚩**:   
   #### Input Options
   - #### Basic text summarization --text
   ```bash
   python summarizer.py --text "Python is an interpreted, high-level programming language..."
   ```
   - #### Summarize from a URL (news/articles) --url
   ```bash
   python summarizer.py --url https://en.wikipedia.org/wiki/Python_(programming_language)
   ```
   - #### Process a local text file --file
   ```bash
   python summarizer.py --file sample.txt
   ```
   - #### Validate training data structure --inspect-training
   ```bash
   python summarizer.py --inspect-training
   ```
   #### Output Options
   - #### Save summary to file --output
   ```bash
   python summarizer.py --text "Long article..." --output summary.txt
   ```
   - #### Bullet-point format --bullets
   ```bash
   python summarizer.py --file sample.txt --bullets
   ```
   - #### Combine output flags
   ```bash
   python summarizer.py --url https://news.com/article --output news_summary.txt --bullets
   ```
   #### Configuration
   - #### Model selection --model (quality | multilingual) default: fast
   ```bash
   python summarizer.py --text "Research paper..." --model quality
   ```
   - #### Summary length --length (short | long) default: medium
   ```bash
   python summarizer.py --file sample.txt --length long
   ```
   - #### Debug mode --debug
   ```bash
   python summarizer.py --text "Debug this" --debug
   ```
   - #### Full configuration example
   ```bash
   python summarizer.py --url https://techblog.com/post --model quality --length short --bullets --output tech_summary.txt --debug
   ```
3. **Project Structure**
   ```text
   ai_summarizer/
   ├── summarizer/
   │   ├── cli.py          # Command line interface
   │   ├── engine.py       # Core processing logic
   │   ├── training.py     # Model training utils
   │   └── hardware/       # Optimization modules
   ├── tests/              # Unit tests
   ├── training_data/      # Example datasets
   ├── requirements.txt    # Dependencies
   └── README.md           # Documentation
   ```

4. **Example Output**
   ```text
      ⚙️ [DEBUG] Initializing quality model on CUDA
   ==================================================
   📝 SUMMARY (42 words)
   ==================================================
   Python is an interpreted, high-level programming language 
   emphasizing code readability. Its design philosophy and 
   syntax enable programmers to express concepts in fewer 
   lines of code than languages like C++ or Java.
   ==================================================
   ⚡ Processed 892 characters in 1.4s (637 chars/sec)
   ```