"""
Self-contained single-file HTML report with embedded CSS/JS.

Includes: pipeline overview, per-agent scorecards, error propagation D3 graph,
bias detection results, cost breakdown, trend charts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from multiagent_eval.core.runner import EvalResult
from multiagent_eval.core.trace import PipelineTrace
from multiagent_eval.judges.propagation_judge import PropagationJudge, PropagationReport


class HTMLReporter:
    """Generates self-contained HTML reports with D3.js visualizations."""

    def __init__(
        self,
        propagation_report: Optional[PropagationReport] = None,
        similarity_fn: Optional[Any] = None,
        judge_model: Optional[str] = None,
    ) -> None:
        """
        Initialize reporter.

        Args:
            propagation_report: Pre-computed propagation report (optional).
            similarity_fn: For computing propagation if not provided.
            judge_model: LLM model for propagation verdict (e.g. ollama/mistral).
        """
        self.propagation_report = propagation_report
        self.similarity_fn = similarity_fn
        self.judge_model = judge_model

    def generate(
        self,
        result: EvalResult,
        output_path: str,
        title: str = "Multi-Agent Evaluation Report",
    ) -> str:
        """
        Generate self-contained HTML report.

        Args:
            result: Evaluation result.
            output_path: Path to write HTML.
            title: Report title.

        Returns:
            Path to generated file.
        """
        propagation = self.propagation_report
        if propagation is None and result.pipeline_trace.agents:
            model = self.judge_model or (getattr(result.config, "judge_primary_model", None) if result.config else None)
            judge = PropagationJudge(similarity_fn=self.similarity_fn, judge_model=model)
            propagation = judge.analyze(result.pipeline_trace)

        html = self._build_html(result, propagation, title)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        return str(path)

    def _build_html(
        self,
        result: EvalResult,
        propagation: Optional[PropagationReport],
        title: str,
    ) -> str:
        """Build full HTML document."""
        from datetime import datetime
        run_timestamp = datetime.now().isoformat()
        pt = result.pipeline_trace
        passed = result.passed()
        total_score = sum(m.score for m in result.metrics) / len(result.metrics) if result.metrics else 0

        metrics_by_agent: dict[str, list] = {}
        pipeline_metrics: list = []
        for m in result.metrics:
            if m.agent_id:
                metrics_by_agent.setdefault(m.agent_id, []).append(m)
            else:
                pipeline_metrics.append(m)

        agent_cards = ""
        for agent in pt.agents:
            ms = metrics_by_agent.get(agent.agent_id, [])
            badges = " ".join(
                f'<span class="badge {"pass" if not m.flagged else "fail"}">{m.metric_name}: {m.score:.2f}</span>'
                for m in ms
            )
            agent_cards += f"""
            <div class="agent-card">
                <h4>{agent.agent_id} ({agent.agent_role})</h4>
                <div class="badges">{badges}</div>
                <p>Latency: {agent.latency_ms:.0f}ms | Cost: ${agent.total_cost_usd():.4f}</p>
                {f'<p class="error">Error: {agent.error}</p>' if agent.error else ''}
            </div>
            """

        graph_data = propagation.graph_data if propagation else {"nodes": [], "links": []}
        graph_json = json.dumps(graph_data)

        bias_section = ""
        for m in result.metrics:
            bias = m.raw_data.get("bias_checks", [])
            if bias:
                for b in bias:
                    bias_section += f'<p><strong>{b.get("bias_type", "?")}</strong>: {b.get("explanation", "")} {"⚠️" if b.get("detected") else "✓"}</p>'

        cost_breakdown = ""
        for agent in pt.agents:
            for call in agent.llm_calls:
                cost_breakdown += f"<tr><td>{agent.agent_id}</td><td>{call.model}</td><td>${call.cost_usd:.4f}</td></tr>"
        if not cost_breakdown:
            cost_breakdown = '<tr><td colspan="3">No LLM calls recorded. Run <code>python examples/langgraph_example.py</code> (with Ollama) for real LLM traces.</td></tr>'

        # Trace flow: input → agents → output
        trace_flow = ""
        if pt.agents:
            first_input = pt.agents[0].input_received
            trace_flow += f'<div class="flow-box"><span class="flow-label">INPUT</span><pre>{json.dumps(first_input, indent=2, ensure_ascii=False)}</pre></div>'
            for agent in pt.agents:
                trace_flow += f'<span class="flow-arrow">→</span>'
                trace_flow += f'<div class="flow-box"><span class="flow-label">{agent.agent_id}</span><pre>{json.dumps(agent.output_produced, indent=2, ensure_ascii=False)[:500]}</pre></div>'
            trace_flow += f'<span class="flow-arrow">→</span><div class="flow-box"><span class="flow-label">FINAL</span><pre>{json.dumps(pt.final_output, indent=2, ensure_ascii=False)}</pre></div>'

        # Golden vs actual
        expected = result.metadata.get("expected_output")
        golden_vs_actual = ""
        if expected:
            actual_str = json.dumps(pt.final_output, indent=2, ensure_ascii=False)
            expected_str = json.dumps(expected, indent=2, ensure_ascii=False)
            golden_vs_actual = f"""
            <div class="comparison-grid">
                <div class="comparison-col">
                    <h4>Expected (Golden)</h4>
                    <pre>{expected_str}</pre>
                </div>
                <div class="comparison-col">
                    <h4>Actual (Produced)</h4>
                    <pre>{actual_str}</pre>
                </div>
            </div>
            """
        else:
            golden_vs_actual = "<p>No golden dataset comparison for this run.</p>"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: system-ui, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
        .overview {{ background: #16213e; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .overview h1 {{ margin-top: 0; }}
        .status {{ font-size: 1.2em; padding: 8px; border-radius: 4px; }}
        .status.pass {{ background: #0f3460; color: #4ecca3; }}
        .status.fail {{ background: #e94560; color: #fff; }}
        .agent-card {{ background: #16213e; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #4ecca3; }}
        .agent-card .error {{ color: #e94560; }}
        .badge {{ display: inline-block; padding: 4px 8px; margin: 2px; border-radius: 4px; font-size: 0.9em; }}
        .badge.pass {{ background: #0f3460; color: #4ecca3; }}
        .badge.fail {{ background: #e94560; color: #fff; }}
        #propagation-graph {{ width: 100%; height: 400px; background: #16213e; border-radius: 8px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #333; }}
        th {{ background: #0f3460; }}
        .section {{ margin: 20px 0; }}
        .section h2 {{ color: #4ecca3; border-bottom: 1px solid #333; padding-bottom: 8px; }}
        .trace-flow {{ display: flex; align-items: flex-start; flex-wrap: wrap; gap: 12px; margin: 16px 0; }}
        .flow-box {{ background: #16213e; padding: 12px; border-radius: 8px; min-width: 200px; max-width: 400px; border-left: 4px solid #4ecca3; }}
        .flow-label {{ color: #7b68ee; font-weight: bold; display: block; margin-bottom: 8px; }}
        .flow-box pre {{ margin: 0; font-size: 12px; overflow-x: auto; white-space: pre-wrap; }}
        .flow-arrow {{ font-size: 1.5em; color: #4ecca3; align-self: center; }}
        .comparison-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .comparison-col {{ background: #16213e; padding: 16px; border-radius: 8px; }}
        .comparison-col h4 {{ margin-top: 0; color: #4ecca3; }}
        .comparison-col pre {{ font-size: 12px; overflow-x: auto; }}
        @media (max-width: 768px) {{ .comparison-grid {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <div class="overview">
        <h1>{title}</h1>
        <p><strong>Run ID:</strong> {pt.pipeline_id} | <strong>Pipeline:</strong> {pt.pipeline_name}</p>
        <p><strong>Executed:</strong> {run_timestamp} | <strong>Agents:</strong> {len(pt.agents)}</p>
        <p><strong>Total Latency:</strong> {pt.total_latency_ms}ms | <strong>Total Cost:</strong> ${pt.total_cost_usd:.4f}</p>
        <p class="status {'pass' if passed else 'fail'}">Overall: {'PASS' if passed else 'FAIL'} (avg score: {total_score:.2%})</p>
    </div>

    <div class="section">
        <h2>Trace Flow</h2>
        <p>Input → Agents → Output</p>
        <div class="trace-flow">{trace_flow}</div>
    </div>

    <div class="section">
        <h2>Expected vs Actual</h2>
        <p>Golden dataset expected output vs pipeline actual output</p>
        {golden_vs_actual}
    </div>

    <div class="section">
        <h2>Per-Agent Scorecards</h2>
        {agent_cards}
    </div>

    <div class="section">
        <h2>Error Propagation Graph</h2>
        <div id="propagation-graph"></div>
    </div>

    <div class="section">
        <h2>Bias Detection</h2>
        {bias_section or '<p>No bias checks in this run.</p>'}
    </div>

    <div class="section">
        <h2>Cost Breakdown</h2>
        <table>
            <tr><th>Agent</th><th>Model</th><th>Cost</th></tr>
            {cost_breakdown or '<tr><td colspan="3">No LLM calls recorded.</td></tr>'}
        </table>
    </div>

    <script>
        const graphData = {graph_json};
        if (graphData.nodes && graphData.nodes.length > 0) {{
            const width = document.getElementById('propagation-graph').clientWidth;
            const height = 400;
            const svg = d3.select('#propagation-graph').append('svg').attr('width', width).attr('height', height);
            const g = svg.append('g');
            const simulation = d3.forceSimulation(graphData.nodes)
                .force('link', d3.forceLink(graphData.links).id(d => d.id).distance(100))
                .force('charge', d3.forceManyBody().strength(-200))
                .force('center', d3.forceCenter(width/2, height/2));
            const link = g.selectAll('line').data(graphData.links).join('line').attr('stroke', '#4ecca3').attr('stroke-width', d => (d.weight || 0.5) * 4);
            const node = g.selectAll('circle').data(graphData.nodes).join('circle').attr('r', 12).attr('fill', d => d.error ? '#e94560' : '#4ecca3').attr('stroke', '#fff').call(d3.drag().on('start', dragstart).on('drag', drag).on('end', dragend));
            const label = g.selectAll('text').data(graphData.nodes).join('text').text(d => d.label || d.id).attr('x', 15).attr('y', 4).attr('fill', '#eee').attr('font-size', 10);
            simulation.on('tick', () => {{ link.attr('x1', d => d.source.x).attr('y1', d => d.source.y).attr('x2', d => d.target.x).attr('y2', d => d.target.y); node.attr('cx', d => d.x).attr('cy', d => d.y); label.attr('x', d => d.x + 15).attr('y', d => d.y + 4); }});
            function dragstart(e) {{ simulation.alphaTarget(0.3).restart(); e.subject.fx = e.subject.x; e.subject.fy = e.subject.y; }}
            function drag(e) {{ e.subject.fx = e.x; e.subject.fy = e.y; }}
            function dragend(e) {{ simulation.alphaTarget(0); e.subject.fx = null; e.subject.fy = null; }}
        }} else {{
            document.getElementById('propagation-graph').innerHTML = '<p style="padding:20px;color:#888">Single agent or no graph data.</p>';
        }}
    </script>
    <p style="margin-top:30px;color:#666"><a href="../docs/view_golden_dataset.html" style="color:#4ecca3">→ View golden dataset</a></p>
</body>
</html>"""
