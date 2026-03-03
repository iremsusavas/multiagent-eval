# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-02-19

### Added

- Initial release
- Propagation-aware evaluation for multi-agent AI systems
- Core metrics: factual_accuracy, error_propagation_score, inter_agent_consistency
- TF-IDF cosine similarity for topic drift detection (no external deps)
- Failure mode taxonomy: PROPAGATION_ERROR, CASCADE_FAILURE, HALLUCINATION, etc.
- LangGraph, CrewAI, AutoGen adapters
- LLM-as-Judge with bias detection (primacy, verbosity, tone, cascade)
- Golden dataset support with schema and annotator
- HTML/JSON reporters, Streamlit dashboard
- OpenTelemetry integration
- PII and prompt injection detection
- CLI: run, estimate-cost, regression-diff, report, dashboard
