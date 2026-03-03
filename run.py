#!/usr/bin/env python3
"""
MultiAgent-Eval - Run everything with one command: python run.py
"""

import subprocess
import sys
from pathlib import Path

# Add src to path
src = Path(__file__).parent / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))


def main() -> None:
    # Optional Ollama check
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and "mistral" not in result.stdout:
            print("Tip: Run 'ollama pull mistral' for Ollama models")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print("\n=== Running evaluation ===\n")
    import sys
    sys.argv = ["multiagent-eval", "run", "--config", "eval_config.yaml"]
    from multiagent_eval.cli import app
    app()

    print("\n=== Done ===")
    print("Results: eval_results/result.json")
    print("Report: eval_results/report.html")


if __name__ == "__main__":
    main()
