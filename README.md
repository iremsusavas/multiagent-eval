# multiagent-eval

> for multi-agent AI systems.

Single-LLM eval tools (RAGAS, DeepEval) miss what actually
breaks in production: errors that start in Agent 1, silently
propagate through Agent 3, and surface in the final output
with no trace of origin.

**multiagent-eval finds where the fault began.**

---

## The Problem Nobody Is Solving

```
Agent 1 ──► Agent 2 ──► Agent 3 ──► Agent 4 (Writer)
   │           │           │            │
   ✓           ✗           ✗            ✗  ← Error propagates; eval sees only final failure
```

You run eval. Score looks good. You ship.

Three days later: a hallucination in production. You check
your eval results. Everything passed.

**What happened?**

Your eval checked the final output. It didn't check whether
Agent 2 silently corrupted the information Agent 1 found.
It didn't check whether Agent 3's hallucination was its own
fault, or the result of broken input from upstream.

That's the gap multiagent-eval closes.

---

## Quickstart

```bash
pip install multiagent-eval
```

```python
from multiagent_eval import evaluate
from multiagent_eval.integrations import LangGraphAdapter

# Option A: Pass adapter (LangGraph, CrewAI, AutoGen)
adapter = LangGraphAdapter(graph=my_langgraph_graph)
result = evaluate(pipeline=adapter, dataset="datasets/my_golden_set.json")

# Option B: Pass PipelineTrace directly
result = evaluate(pipeline=my_trace, ground_truth={"expected": "..."})

print(result.overall_score)   # 0.84
print(result.passed())       # True / False
```

---

## What Makes This Different

### Propagation Judge
Detects where information corruption begins — not just
that it happened. Builds a directed graph where each edge
carries a fidelity score. Red edges show exactly where
data was lost or distorted between agents.

### Built-in Bias Detection
Every LLM judge call automatically runs:
- **Primacy bias** (A/B swap permutation tests)
- **Verbosity bias** (length vs. correctness)
- **Tone bias** (neutral vs. apologetic framing)
- **Cascade bias** (upstream error penalizing innocent agents)

### CI/CD Native
Eval isn't a report. It's a gate.

```yaml
# .github/workflows/multiagent-eval.yml
- name: Run evaluation
  run: multiagent-eval run --config eval_config.yaml
# eval_score < threshold → fail the PR
```

### Statistical Rigor
Bootstrap confidence intervals and permutation p-values
on every run. "Did we improve?" becomes answerable.

### Failure Mode Taxonomy
Not just a score. A category:

`PROPAGATION_ERROR` | `HALLUCINATION` | `CONTEXT_LOSS` |
`ORCHESTRATION_BREAK` | `CASCADE_FAILURE` | `PII_LEAKAGE`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         multiagent-eval                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  core/           trace, metrics, runner, state_machine, LLMGateway        │
│  judges/         LLMJudge (CoT, bias), ConsistencyJudge, PropagationJudge│
│  bias_detection/ primacy, verbosity, tone, cascade                       │
│  golden_datasets/ schema, manager, annotator, inter-rater agreement       │
│  reports/        JSON, HTML (D3.js), Streamlit dashboard                  │
│  integrations/   LangGraph, CrewAI, AutoGen, Custom adapters               │
│  telemetry/      OpenTelemetry spans → Datadog, Grafana, Jaeger          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Integrations

| LangGraph | CrewAI | AutoGen | Custom |

---

## Production Features

- **OpenTelemetry**: Real-time span emission to Datadog/Grafana/Jaeger
- **PII Detection**: Email, SSN, credit card — zero-tolerance config
- **Prompt Injection Detection**: Pattern-based, extensible
- **Cost Estimation**: Know your budget before you run (`estimate-cost --dataset ...`)
- **Regression Testing**: Which examples degraded between v1.1 and v1.2? (`regression-diff`)

---

## CLI

```bash
multiagent-eval run --config eval_config.yaml
multiagent-eval run --all                    # All examples in golden dataset
multiagent-eval estimate-cost -d datasets/research_qa.json
multiagent-eval regression-diff -a result_v1.json -b result_v2.json
multiagent-eval report --input results.json --format html
multiagent-eval dataset add --name my_dataset
multiagent-eval dashboard
```

---

## Background

Built by an ML engineer who spent months improving
LLM-as-Judge agreement from 63% to 84% in production
at Pipedrive — and discovered that most eval problems
aren't scoring problems. They're architectural ones.

JudgeGuard (primacy bias detection) came first.
multiagent-eval is what came after asking:
*"What happens to these biases when you have five agents?"*

---

## Roadmap

- [ ] Leaderboard / MAE-Bench public benchmark
- [ ] Multi-turn stateful session evaluation
- [ ] Visual diff UI for agent output comparison
- [ ] Automated rubric improvement suggestions
- [ ] Native LangSmith integration

---

## Contributing

Issues, PRs, and dataset contributions welcome.
If you're building multi-agent systems and hitting
eval problems — open an issue. That's how this gets better.

---

## License

MIT
