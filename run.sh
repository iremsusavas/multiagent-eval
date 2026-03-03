#!/bin/bash
# MultiAgent-Eval - Run everything with one command

set -e
cd "$(dirname "$0")"

# Ollama check (for free local models)
if command -v ollama &> /dev/null; then
    if ! ollama list 2>/dev/null | grep -q mistral; then
        echo "Downloading Mistral model (may take a few minutes on first run)..."
        ollama run mistral --help 2>/dev/null || ollama pull mistral
    fi
else
    echo "Note: Ollama not installed. For free models: https://ollama.ai"
    echo "For Groq: set primary_model: groq/llama-3.1-70b-versatile in eval_config.yaml"
fi

# PYTHONPATH (if pip install was not run)
export PYTHONPATH="${PYTHONPATH:-}:${PWD}/src"

echo ""
echo "=== Running evaluation ==="
python -m multiagent_eval.cli run --config eval_config.yaml

echo ""
echo "=== Done ==="
echo "Results: eval_results/result.json"
echo "Report: eval_results/report.html"
echo ""
echo "To open the report: open eval_results/report.html"
