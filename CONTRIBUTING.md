# Contributing to multiagent-eval

Thank you for your interest in contributing.

## How to contribute

- **Issues**: Found a bug or have a feature idea? Open an issue.
- **Pull requests**: Fixes and improvements welcome. Please run tests before submitting.
- **Datasets**: Golden dataset contributions (research_qa format) are especially valuable.

## Development setup

```bash
git clone https://github.com/iremsusavas/multiagent-eval.git
cd multiagent-eval
pip install -e ".[dev]"
pytest tests/ -v
```

## Code style

- Ruff for linting: `ruff check src/`
- Type hints encouraged for public APIs

## If you're building multi-agent systems

If you're hitting eval problems in production — open an issue. That's how this gets better.
