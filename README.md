# Hyperbolic: AI Machine-Health Assistant

Built for the **AMD Developer Hackathon: ACT III** (Intelligent Industry track).

An AI assistant that spots equipment problems early and tells technicians what to do next, in plain language.

> Status: work in progress (hackathon build, 12-18 October 2026).

## What it does

1. **Early warning:** a predictive model analyzes machine sensor data and flags abnormal patterns before a failure.
2. **Plain-language explanation:** an open-source language model, served on AMD GPUs, explains each alert and recommends a maintenance action.
3. **Manual search:** technicians ask questions about maintenance procedures and get answers with sources.

## Tech stack

- Python
- AMD Developer Cloud (AMD Instinct GPUs) with ROCm
- vLLM serving an open-source language model
- scikit-learn for the failure-detection model
- Gradio for the web interface

## Setup

```bash
git clone https://github.com/mohidcodes/hyperbolic-machine-health.git
cd hyperbolic-machine-health
pip install -r requirements.txt
cp .env.example .env   # then fill in your own values
```

Never commit your `.env` file. It holds private keys.

## Team

Hyperbolic

## License

MIT
