# Examples

## Quickstart (no API key needed)

```bash
git clone https://github.com/iremsusavas/multiagent-eval.git
cd multiagent-eval
pip install -e .

python examples/quickstart_mock.py
```

This runs a mock 3-agent pipeline (Researcher → Analyst → Writer),
injects a hallucination in Agent 2, and shows how multiagent-eval
traces the fault origin via the Propagation Judge.

Expected output:

```
HEALTHY PIPELINE
  Overall Score : 0.82+
  Passed        : True
  Per-Agent Scores: agent_001 ~0.91, agent_002 ~0.85, agent_003 ~0.78
  Failure Modes : ✓ None detected

CORRUPTED PIPELINE (Agent 2 hallucinates — switches to climate change)
  Overall Score : 0.35 or below
  Passed        : False
  Per-Agent Scores: agent_002: 0.00  ← fault origin
  Failure Modes : PROPAGATION_ERROR (agent_002 → agent_003), CASCADE_FAILURE
```

## LangGraph Integration

```bash
pip install -e . langgraph openai
export OPENAI_API_KEY=sk-...
python examples/quickstart_langgraph.py
```
