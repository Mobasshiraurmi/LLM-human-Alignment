"""report_replay.py — one HTML report for any step5_replay.py --agent run.

Generalized from archive/report_2A-C1.py (which was multi-agent-only) to work for
any registered agent (see step5_replay.AGENTS): pass --agent {single,multi} and it
reads entirely from replay/<agent>/, written by step5_replay.py + classify_replay.py.

Report first, then logs:
  1. Divergence table: this architecture vs. the human baseline, via
     kodis_stats.py's corrected divergence-gap engine -- the same statistics
     used by report/kodis_reports.py's 4 comparison reports (the paper's actual
     statistical test; there is no separate test here).
  2. Logs: the full conversation table (human turn beside model turn, per-turn
     action/burst size), regenerated from whatever is currently in
     replay/<agent>/replay.jsonl and replay/<agent>/results.csv.

Self-contained: writes replay/<agent>/report.html and opens it. Re-run anytime
after a new step5_replay.py run + classify_replay.py pass to regenerate both
sections from the current data.

    python report/report_replay.py --agent single
    python report/report_replay.py --agent multi
"""
import argparse
import json
import os
import subprocess
import sys
import webbrowser

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "stats"))        # rag_align, kodis_stats
sys.path.insert(0, os.path.join(_ROOT, "generation"))   # step5_replay (kodis_stats.human_baseline_47 imports it)
import rag_align as ra
import kodis_stats as k

HUMAN_CSV = "data/human_data.csv"   # same file the model is evaluated against
                                                    # at generation time (step5_replay.HUMAN_CSV)

AGENT_LABEL = {
    "single": "single-agent (one direct system-prompted call, step5_replay.py --agent single)",
    "multi": "multi-agent (two_agents.py planner+realizer, step5_replay.py --agent multi)",
    "single_personality": "single-agent + Big-Five personality (Kwon et al., EMNLP 2025 -- "
                          "see personality.py), step5_replay.py --agent single_personality",
    "single_rag": "single-agent + RAG (state-matched real human turns as exemplars -- "
                  "see single_agent.py/rag_align.py), step5_replay.py --agent single_rag",
    "single_claude-sonnet-4.6": "single-agent, generating LLM Claude Sonnet 4.6 (llm_backends.py)",
    "single_gemini-2.5-pro": "single-agent, generating LLM Gemini 2.5 Pro (llm_backends.py)",
    "single_personality_claude-sonnet-4.6": "single-agent + Big-Five personality, generating LLM Claude Sonnet 4.6",
    "single_personality_gemini-2.5-pro": "single-agent + Big-Five personality, generating LLM Gemini 2.5 Pro",
    "multi_claude-sonnet-4.6": "multi-agent (two_agents.py planner+realizer), generating LLM Claude Sonnet 4.6",
    "multi_gemini-2.5-pro": "multi-agent (two_agents.py planner+realizer), generating LLM Gemini 2.5 Pro",
}


# ---------------------------------------------------------------------------
# 1. Load + compute the divergence table (kodis_stats.py)
# ---------------------------------------------------------------------------

def load_sim(results_csv):
    res = pd.read_csv(results_csv)
    res["Behavior"] = ra._canon_behavior(res["pred_behavior"])
    res["ProlificId"] = res["Sender"]
    res["Role"] = res["Sender"]
    return res.dropna(subset=["Behavior"])[
        ["GameId", "ProlificId", "Role", "Round", "Phase", "Text", "Behavior"]
    ]


def kodis_section(sim: pd.DataFrame, agent_label: str):
    """The paper's actual statistical test: kodis_stats.py's corrected divergence-gap
    engine (permutation of dyads, not a t-test over dependent pairwise values -- see
    that module's docstring), one row for this architecture against the SAME fixed
    47-dyad human baseline every report in this repo uses (k.human_baseline_47 --
    the dyads with an identifiable BATNA recipient; not all 50 raw human dyads)."""
    human_dyads = k.human_baseline_47(HUMAN_CSV)
    agent_dyads = k.per_dyad(sim)
    agents = {agent_label: agent_dyads}
    paired = {agent_label: set(agent_dyads) <= set(human_dyads)}
    t1 = k.run_table(k.TABLE1, human_dyads, agents, paired)
    t2 = k.run_table(k.TABLE2, human_dyads, agents, paired)
    n_h = len(human_dyads)
    cap1 = f"Table 1: Divergence metric results for {agent_label} compared to human negotiators."
    cap2 = f"Table 2: Behavior-share gaps by phase for {agent_label} compared to human negotiators."
    html = (k.method_note(n_h, HUMAN_CSV)
            + k.html_table(k.TABLE1, t1, agents, cap1, n_h)
            + k.html_dispersion(k.TABLE1, t1, agents, "Table 1b: Within-group dispersion for the metrics of Table 1.")
            + k.html_table(k.TABLE2, t2, agents, cap2, n_h)
            + k.html_dispersion(k.TABLE2, t2, agents, "Table 2b: Within-group dispersion for the metrics of Table 2."))
    tex = (k.tex_table(k.TABLE1, t1, agents, cap1, n_h) + "\n\n"
          + k.tex_table(k.TABLE2, t2, agents, cap2, n_h) + "\n")
    return html, tex


# ---------------------------------------------------------------------------
# 2. The conversation log table
# ---------------------------------------------------------------------------

def _model_behaviors_by_turn(unlabeled_csv, results_csv) -> dict:
    """(game_id, turn_idx) -> ordered list of pred_behavior, one per model message.

    results.csv (classify_replay.py's output) has no turn_idx/message_index of its own --
    it's clf_input.csv (a column subset of replay_unlabeled.csv, same row order) with
    pred_* columns appended in place. So it lines up positionally with
    replay_unlabeled.csv, which IS sorted by (GameId, turn_idx, message_index) -- same
    positional-join trick step6_paired.load_pairs already relies on for this file pair.
    """
    unlab = pd.read_csv(unlabeled_csv)
    res = pd.read_csv(results_csv)
    if len(unlab) != len(res):
        raise ValueError(f"{unlabeled_csv} has {len(unlab)} rows but {results_csv} has {len(res)} -- "
                         "some classifier requests failed, or results.csv is stale. "
                         "Re-run classify_replay.py --parse.")
    out: dict = {}
    for gid, tidx, beh in zip(unlab["GameId"], unlab["turn_idx"], res["pred_behavior"]):
        out.setdefault((gid, tidx), []).append(beh if pd.notna(beh) else "?")
    return out


def build_log_table(jsonl, traces_jsonl, unlabeled_csv, results_csv) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(jsonl)]
    traces = {}
    if os.path.exists(traces_jsonl):
        for l in open(traces_jsonl):
            t = json.loads(l)
            traces[(t["game_id"], t["turn_idx"])] = t
    model_behaviors = _model_behaviors_by_turn(unlabeled_csv, results_csv)
    df = pd.DataFrame(rows)
    # "action" comes from a planner decision -- only the multi-agent responder makes one.
    # For single-agent traces (no "planner_decision" key) this stays blank, which is the
    # correct way to render "not applicable to this architecture", not a bug.
    df["action"] = df.apply(
        lambda r: traces.get((r.game_id, r.turn_idx), {}).get("planner_decision", {}).get("action", ""), axis=1)
    df["human_side"] = df.apply(
        lambda r: "<br>".join(f"[{b}] {m}" for b, m in zip(r.human_behaviors, r.human_messages)), axis=1)
    df["model_side"] = df.apply(
        lambda r: "<br>".join(f"[{b}] {m}" for b, m in zip(
            model_behaviors.get((r.game_id, r.turn_idx), []), r.model_messages)), axis=1)
    cols = ["game_id", "turn_idx", "round", "phase", "role", "is_recipient", "action",
            "burst_size", "model_burst_size", "human_side", "model_side"]
    return df.sort_values(["game_id", "turn_idx"])[cols].rename(
        columns={"burst_size": "human_burst", "model_burst_size": "model_burst"})


# ---------------------------------------------------------------------------
# 3. Assemble
# ---------------------------------------------------------------------------

def main(agent: str, out_dir: str = "data/replay"):
    run_dir = f"{out_dir}/{agent}"
    results_csv = f"{run_dir}/results.csv"
    jsonl = f"{run_dir}/replay.jsonl"
    unlabeled_csv = f"{run_dir}/replay_unlabeled.csv"
    traces_jsonl = f"{run_dir}/private_traces.jsonl"
    out_html = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"{run_dir}/report.html")

    for f in (jsonl, unlabeled_csv, results_csv):
        if not os.path.exists(f):
            sys.exit(f"{f} not found. Run generation/step5_replay.py --agent {agent} --run, then "
                     f"classify/classify_replay.py --agent {agent} --submit/--check/--parse first.")

    print("loading data + computing the kodis-style divergence table (this takes a minute)...")
    sim = load_sim(results_csv)
    n_dyads = sim.GameId.nunique()
    agent_label = AGENT_LABEL.get(agent, agent)
    kodis_html, kodis_tex = kodis_section(sim, agent_label)
    tex_path = f"{run_dir}/alignment_table.tex"
    with open(tex_path, "w") as fh:
        fh.write(kodis_tex)
    print(f"wrote {tex_path}")
    log_df = build_log_table(jsonl, traces_jsonl, f"{run_dir}/replay_unlabeled.csv", results_csv)
    log_table_html = log_df.to_html(index=False, na_rep="", escape=False, classes="data")

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{agent}-agent replay report</title>
<style>
{k.CSS}
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #0b0b0b; background: #fcfcfb; }}
h1 {{ font-size: 20px; margin-bottom: 4px; }}
h2 {{ font-size: 15px; margin-top: 36px; border-bottom: 1px solid #dedcd3; padding-bottom: 6px; }}
.subtitle {{ color: #52514e; font-size: 13px; margin-bottom: 24px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin-bottom: 8px; }}
th, td {{ border: 1px solid #dedcd3; padding: 6px 8px; text-align: left; vertical-align: top; }}
th {{ background: #2c3e50; color: white; position: sticky; top: 0; }}
tr:nth-child(even) {{ background: #f7f7f5; }}
.log-table td {{ max-width: 260px; white-space: normal; word-wrap: break-word; }}
.note {{ font-size: 12px; color: #52514e; margin-top: 8px; }}
</style></head><body>

<h1>{agent}-agent replay — alignment report</h1>
<div class="subtitle">{agent_label}, {n_dyads} dyads, {len(sim)} model messages
(from {results_csv} / {jsonl})</div>

<h2>Divergence from human negotiators (kodis_stats.py)</h2>
{kodis_html}

<h2>Conversation logs — all {n_dyads} dyads</h2>
<div class="note">One row per turn. human_side/model_side show every message in that turn's burst,
each tagged with its behavior label: [behavior] text. Human tags are the corpus's own labels;
model tags are classify_replay.py's per-message predictions (pred_behavior).
"action" is the multi-agent planner's decision for that turn and is blank for single-agent runs,
which have no separate planning step.</div>
{log_table_html}

</body></html>"""

    os.makedirs(run_dir, exist_ok=True)
    with open(out_html, "w") as f:
        f.write(html)
    print(f"wrote {out_html}")
    try:
        subprocess.run(["open", out_html], check=False)
    except Exception:
        webbrowser.open(f"file://{out_html}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=["single", "multi", "single_personality", "single_rag", "single_claude-sonnet-4.6", "single_gemini-2.5-pro", "single_personality_claude-sonnet-4.6", "single_personality_gemini-2.5-pro", "multi_claude-sonnet-4.6", "multi_gemini-2.5-pro"],
                    help="which step5_replay.py run to report on (replay/<agent>/).")
    ap.add_argument("--out-dir", default="data/replay",
                    help="base directory to report on (default: data/replay, the shipped results).")
    a = ap.parse_args()
    main(a.agent, a.out_dir)
