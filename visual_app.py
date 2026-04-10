import os
import re
from typing import Any

import gradio as gr
import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from agents.rule_agent import choose_action, estimate_risk
from env.environment import SOCEnv


STAGE_LABELS = {
    "normal_activity": "Benign Activity",
    "initial_access": "Initial Access",
    "privilege_escalation": "Privilege Escalation",
    "credential_access": "Lateral Movement",
    "lateral_movement": "Lateral Movement",
    "exfiltration": "Data Exfiltration",
}

ALERT_SUMMARIES = {
    "normal_activity": "Routine user activity",
    "initial_access": "Suspicious login",
    "privilege_escalation": "Privilege escalation",
    "credential_access": "Credential dumping / lateral prep",
    "lateral_movement": "Remote service movement",
    "exfiltration": "Potential data exfiltration",
}

MITRE_MAP = {
    "initial_access": ("T1078", "Valid Accounts"),
    "privilege_escalation": ("T1068", "Exploitation for Privilege Escalation"),
    "credential_access": ("T1003", "OS Credential Dumping"),
    "lateral_movement": ("T1021", "Remote Services"),
    "exfiltration": ("T1041", "Exfiltration Over C2 Channel"),
}

LEADERBOARD_ROWS = [
    {"Agent": "Baseline Rule Agent", "Score": 0.72},
    {"Agent": "GPT-Agent", "Score": 0.85},
    {"Agent": "Human Analyst", "Score": 0.95},
]


def _stage_label(stage: str) -> str:
    return STAGE_LABELS.get(stage, stage.replace("_", " ").title())


def _alert_summary(alert: dict[str, Any]) -> str:
    stage = alert.get("stage", "unknown")
    return ALERT_SUMMARIES.get(stage, f"Alert related to {_stage_label(stage)}")


def _mitre_for_stage(stage: str) -> tuple[str, str]:
    return MITRE_MAP.get(stage, ("T1595", "Active Scanning"))


def _extract_reason_lines(alert: dict[str, Any], risk: int) -> list[str]:
    reasons = [f"Risk score is {risk}/100 based on severity and stage signals."]
    if alert.get("failed_logins", 0) >= 8:
        reasons.append("Multiple failed logins indicate possible brute-force behavior.")
    if alert.get("new_admin_session"):
        reasons.append("New administrative session suggests privilege escalation.")
    if alert.get("process_name") == "procdump.exe":
        reasons.append("Credential-dumping tooling pattern detected (procdump.exe).")
    if alert.get("bytes_out", 0) > 10000000:
        reasons.append("Large outbound transfer volume indicates potential exfiltration.")
    if alert.get("dest_reputation") == "unknown":
        reasons.append("Destination reputation is unknown and increases investigation priority.")
    if alert.get("src_geo") == "rare":
        reasons.append("Source geography is rare for this environment.")
    if len(reasons) == 1:
        reasons.append("No strong malicious indicators found; likely low-priority activity.")
    return reasons


def _alerts_to_df(alerts: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for i, alert in enumerate(alerts):
        rows.append(
            {
                "alert": f"Alert {i + 1}",
                "event_id": alert.get("event_id"),
                "stage": _stage_label(alert.get("stage", "unknown")),
                "summary": _alert_summary(alert),
                "severity": alert.get("severity", "unknown"),
                "malicious": bool(alert.get("malicious", False)),
            }
        )
    return pd.DataFrame(rows)


def _mitre_df(alerts: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for i, alert in enumerate(alerts):
        stage = alert.get("stage", "unknown")
        tid, name = _mitre_for_stage(stage)
        rows.append(
            {
                "alert": f"Alert {i + 1}",
                "stage": _stage_label(stage),
                "technique": tid,
                "name": name,
            }
        )
    return pd.DataFrame(rows)


def _chain_text(alerts: list[dict[str, Any]]) -> str:
    if not alerts:
        return "No alerts yet. Start a scenario or upload logs."
    nodes = [f"Alert {i + 1} -> {_alert_summary(alert)}" for i, alert in enumerate(alerts)]
    return "\n".join(nodes)


def _chain_figure(alerts: list[dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    if not alerts:
        fig.update_layout(title="Attack Chain Timeline", template="plotly_white")
        return fig

    graph = nx.DiGraph()
    for idx, alert in enumerate(alerts):
        graph.add_node(idx, label=f"Alert {idx + 1}")
        if idx > 0:
            graph.add_edge(idx - 1, idx)

    pos = {idx: (idx, 0) for idx in graph.nodes}

    edge_x = []
    edge_y = []
    for src, dst in graph.edges:
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=2),
            hoverinfo="none",
            showlegend=False,
        )
    )

    node_x = [pos[idx][0] for idx in graph.nodes]
    node_y = [pos[idx][1] for idx in graph.nodes]
    texts = [
        f"Alert {i + 1}<br>{_stage_label(alert.get('stage', 'unknown'))}<br>{_alert_summary(alert)}"
        for i, alert in enumerate(alerts)
    ]

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[f"Alert {i + 1}" for i in range(len(alerts))],
            textposition="top center",
            marker=dict(size=28, color="#1f77b4"),
            hovertext=texts,
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Attack Chain Timeline",
        template="plotly_white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=50, b=20),
        height=300,
    )
    return fig


def _parse_uploaded_lines(lines: list[str]) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for line in lines:
        text = line.strip().lower()
        if not text:
            continue

        event_id_match = re.search(r"event\s*id\s*[:=]?\s*(\d+)", text)
        event_id = int(event_id_match.group(1)) if event_id_match else None

        if "4625" in text or "failed login" in text or "invalid password" in text:
            alerts.append({"event_id": event_id or 4625, "stage": "initial_access", "severity": "high", "malicious": True, "failed_logins": 12, "src_geo": "rare"})
        elif "4672" in text or "privilege escalation" in text or "admin rights" in text:
            alerts.append({"event_id": event_id or 4672, "stage": "privilege_escalation", "severity": "high", "malicious": True, "new_admin_session": True})
        elif "sysmon" in text and ("procdump" in text or "event id 1" in text):
            alerts.append({"event_id": event_id or 1, "stage": "credential_access", "severity": "medium", "malicious": True, "process_name": "procdump.exe"})
        elif "lateral movement" in text or "remote service" in text or "psexec" in text:
            alerts.append({"event_id": event_id or 3, "stage": "lateral_movement", "severity": "medium", "malicious": True})
        elif "5156" in text or "exfil" in text or "bytes out" in text or "upload" in text:
            alerts.append({"event_id": event_id or 5156, "stage": "exfiltration", "severity": "high", "malicious": True, "bytes_out": 25000000, "dest_reputation": "unknown"})
        elif "apache" in text and ("wp-login" in text or "401" in text or "403" in text):
            alerts.append({"event_id": event_id or 10001, "stage": "initial_access", "severity": "medium", "malicious": True, "failed_logins": 10})

    return alerts


def _parse_file(upload: Any) -> list[dict[str, Any]]:
    if upload is None:
        return []

    path = getattr(upload, "name", None)
    if isinstance(upload, dict):
        path = upload.get("name", path)

    if not path or not os.path.exists(path):
        return []

    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        alerts = []
        for _, row in df.iterrows():
            stage = str(row.get("stage", "initial_access")).strip().lower()
            event_id = int(row.get("event_id", 4625)) if not pd.isna(row.get("event_id", None)) else 4625
            severity = str(row.get("severity", "medium")).strip().lower()
            alerts.append(
                {
                    "event_id": event_id,
                    "stage": stage,
                    "severity": severity,
                    "malicious": bool(row.get("malicious", True)),
                    "failed_logins": int(row.get("failed_logins", 0) or 0),
                    "bytes_out": int(row.get("bytes_out", 0) or 0),
                    "dest_reputation": str(row.get("dest_reputation", "unknown")),
                }
            )
        return alerts

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return _parse_uploaded_lines(f.readlines())


def _leaderboard_df() -> pd.DataFrame:
    return pd.DataFrame(LEADERBOARD_ROWS)


def _render_state(state: dict[str, Any]) -> tuple[str, pd.DataFrame, str, go.Figure, pd.DataFrame, str, pd.DataFrame]:
    seen_alerts = [h["alert"] for h in state.get("history", [])]
    current_obs = state.get("current", {}).get("observation")
    if current_obs:
        seen_alerts = seen_alerts + [current_obs]

    status = f"Episode Step: {state['env'].index} / {len(state['env'].chain)}"
    alert_df = _alerts_to_df(seen_alerts)
    chain_text = _chain_text(seen_alerts)
    chain_fig = _chain_figure(seen_alerts)
    mitre = _mitre_df(seen_alerts)

    if current_obs:
        action, _ = choose_action(current_obs, return_explanation=True)
        risk = estimate_risk(current_obs)
        reasons = _extract_reason_lines(current_obs, risk)
        reasoning = "\n".join(
            [
                f"Agent Action: {action.title()}",
                "Reason:",
                *[f"- {line}" for line in reasons],
            ]
        )
    else:
        reasoning = "Incident completed. No active alerts remain."

    return status, alert_df, chain_text, chain_fig, mitre, reasoning, _leaderboard_df()


def start_incident(profile: str, stochastic: bool, episode_length: int) -> tuple[dict[str, Any], str, pd.DataFrame, str, go.Figure, pd.DataFrame, str, pd.DataFrame]:
    env = SOCEnv(stochastic=stochastic, profile=profile, episode_length=episode_length)
    current = env.reset()
    state = {"env": env, "current": current, "history": []}
    rendered = _render_state(state)
    return state, *rendered


def run_next_step(state: dict[str, Any]) -> tuple[dict[str, Any], str, pd.DataFrame, str, go.Figure, pd.DataFrame, str, pd.DataFrame]:
    if not state or not state.get("current"):
        empty = (
            "No scenario loaded.",
            pd.DataFrame(),
            "",
            _chain_figure([]),
            pd.DataFrame(),
            "Start an incident first.",
            _leaderboard_df(),
        )
        return {"env": SOCEnv(), "current": None, "history": []}, *empty

    obs = state["current"].get("observation")
    if obs is None:
        rendered = _render_state(state)
        return state, *rendered

    action, _ = choose_action(obs, return_explanation=True)
    result = state["env"].step(action)
    state["history"].append({"alert": obs, "action": action})
    state["current"] = result
    rendered = _render_state(state)
    return state, *rendered


def analyze_uploaded_logs(upload: Any) -> tuple[pd.DataFrame, str, go.Figure, pd.DataFrame, str, pd.DataFrame]:
    alerts = _parse_file(upload)
    if not alerts:
        return (
            pd.DataFrame(),
            "No recognizable log patterns found. Upload Windows/Sysmon/Apache logs or CSV.",
            _chain_figure([]),
            pd.DataFrame(),
            "No agent recommendation available.",
            _leaderboard_df(),
        )

    first = alerts[0]
    action, _ = choose_action(first, return_explanation=True)
    risk = estimate_risk(first)
    reasons = _extract_reason_lines(first, risk)
    recommendation = "\n".join(
        [
            f"Agent Action: {action.title()}",
            "Reason:",
            *[f"- {line}" for line in reasons],
        ]
    )

    return (
        _alerts_to_df(alerts),
        _chain_text(alerts),
        _chain_figure(alerts),
        _mitre_df(alerts),
        recommendation,
        _leaderboard_df(),
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="SOC-AgentLab Visual Investigation", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
# SOC-AgentLab: Visual SOC Investigation

### Attack Chain
Initial Access -> Privilege Escalation -> Lateral Movement -> Data Exfiltration

This interface adds visual explainability, MITRE ATT&CK mapping, agent reasoning, and multi-stage scenario progression.
            """.strip()
        )

        sim_state = gr.State({"env": SOCEnv(), "current": None, "history": []})

        with gr.Tab("Incident Simulator"):
            with gr.Row():
                profile = gr.Dropdown(
                    label="Adversary Profile",
                    choices=["balanced", "aggressive", "stealthy"],
                    value="balanced",
                )
                stochastic = gr.Checkbox(label="Stochastic Episode", value=True)
                episode_length = gr.Slider(label="Episode Length", minimum=3, maximum=10, value=5, step=1)

            with gr.Row():
                start_btn = gr.Button("Start Multi-Stage Incident", variant="primary")
                next_btn = gr.Button("Run Next Agent Step")

            status = gr.Textbox(label="Scenario Status")
            current_alerts = gr.Dataframe(label="Alert Timeline")
            chain_text = gr.Textbox(label="Attack Timeline Narrative", lines=6)
            chain_plot = gr.Plot(label="Attack Chain Visualization")
            mitre_table = gr.Dataframe(label="MITRE ATT&CK Mapping")
            reasoning = gr.Textbox(label="Agent Reasoning Output", lines=8)
            leaderboard = gr.Dataframe(label="Leaderboard")

            start_btn.click(
                fn=start_incident,
                inputs=[profile, stochastic, episode_length],
                outputs=[sim_state, status, current_alerts, chain_text, chain_plot, mitre_table, reasoning, leaderboard],
            )

            next_btn.click(
                fn=run_next_step,
                inputs=[sim_state],
                outputs=[sim_state, status, current_alerts, chain_text, chain_plot, mitre_table, reasoning, leaderboard],
            )

        with gr.Tab("Real Log Format Support"):
            gr.Markdown(
                "Upload Windows Event Logs, Sysmon logs, Apache logs, or a CSV file with event metadata."
            )
            upload = gr.File(label="Upload Log File")
            parsed_alerts = gr.Dataframe(label="Parsed Alerts")
            parsed_timeline = gr.Textbox(label="Timeline Narrative", lines=6)
            parsed_plot = gr.Plot(label="Parsed Attack Chain")
            parsed_mitre = gr.Dataframe(label="MITRE ATT&CK Mapping")
            parsed_reasoning = gr.Textbox(label="Agent Recommendation", lines=8)
            parsed_leaderboard = gr.Dataframe(label="Benchmark Leaderboard")

            upload.change(
                fn=analyze_uploaded_logs,
                inputs=[upload],
                outputs=[parsed_alerts, parsed_timeline, parsed_plot, parsed_mitre, parsed_reasoning, parsed_leaderboard],
            )

    return demo
