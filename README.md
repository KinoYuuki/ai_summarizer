# AI Summarizer 🔍→📝

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

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
2. **Input**:   
   #### Reading from text
   ```bash
   python summarizer.py --text "Your long text here..."
   ```
   
   #### Reading from URL
   ```bash
   python summarizer.py --url https://example.com/article
   ```

   #### Reading from file
   ```bash
   python summarizer.py --file "long_text.txt"
   ```
   #### Inspecting a train data
   ```bash
   python summarizer.py --train-data
   ```