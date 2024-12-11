# Encoding Attacks

Research implementation for testing LLM robustness through advanced encoding techniques. This project explores various cipher-based approaches to test language model behavior and safety mechanisms.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Add API key to `.env`:
```
ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

### Basic Cipher Attack
Run cipher-based attacks with default parameters:
```bash
python n_cypers.py
```

### Advanced Execution
For more control over execution parameters:
```bash
python run.py --model claude-3-5-sonnet-20241022
```
