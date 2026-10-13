"""report_freerun.py — one HTML report for any step2_generate.py --agent run.

The free-running counterpart to report_replay.py: same divergence statistics (they
operate on generic human-vs-model dataframes, agnostic to how the model data was
generated), but a genuinely different "logs" section -- free-running dialogues are
NOT paired against a specific human turn the way replay's are (each dyad is a whole
invented negotiation, compared only in aggregate to the human corpus), so there is no
replay_pairs.csv equivalent. The log here shows each dyad's own generated conversation
with its predicted behavior tags, plus deal/outcome stats where the architecture
tracks them (multi-agent only -- see two_agents.simulate_multi_agent_dyad).

Report first, then logs:
  1. Divergence table: this architecture vs. the human baseline, via
     kodis_stats.py's corrected divergence-gap engine -- the same statistics
     used by report/kodis_reports.py's 4 comparison reports (the paper's actual
     statistical test; there is no separate test here). Unpaired here (free-running
     dyads have no matching human GameId), so dyad-level metrics fall back to a
     Welch t-test, same as the comparison reports' freerun rows.
  2. Logs: one conversation per dyad, model's own turns tagged with predicted
     behavior, regenerated from whatever is currently in freerun/<agent>/run.jsonl
     and freerun/<agent>/results.csv.

Self-contained: writes freerun/<agent>/report.html and opens it. Re-run anytime
after a new step2_generate.py run + classify_freerun.py pass to regenerate both
sections from the current data.

    python report/report_freerun.py --agent single
    python report/report_freerun.py --agent multi
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
                                                    # at generation time (step2_generate.HUMAN_CSV)

AGENT_LABEL = {
    "single": "single-agent (one direct system-prompted call, step2_generate.py --agent single)",
    "multi": "multi-agent (two_agents.py planner+realizer, full dialogue state, "
             "step2_generate.py --agent multi)",
    "single_personality": "single-agent + Big-Five personality (Kwon et al., EMNLP 2025 -- "
                          "see personality.py), step2_generate.py --agent single_personality",
    "single_rag": "single-agent + RAG (state-matched real human turns as exemplars -- "
                  "see single_agent.py/rag_align.py), step2_generate.py --agent single_rag",
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
# 2. The conversation log table -- one row per DYAD, not per turn. Free-running
#    dialogues aren't paired against a specific human turn the way replay's
#    are (see module docstring), so there's no human_side/model_side split --
#    just the model's own full invented conversation, each message tagged
#    with its predicted behavior.
# ---------------------------------------------------------------------------

def build_dialogue_log(jsonl, results_csv) -> pd.DataFrame:
    """data/freerun/<agent>/run_unlabeled.csv and results.csv are written/read in the
    same flattened (dyad, turn, message) order step2_generate.export() builds them
    in -- for d in dialogues: for t in d["turns"] -- so re-flattening run.jsonl's
    dialogues the same way and zipping positionally against results.csv recovers
    each message's predicted behavior with no join key needed. Same trick
    report_replay.py uses for replay_unlabeled.csv/results.csv.
    """
    dial = [json.loads(l) for l in open(jsonl)]
    res = pd.read_csv(results_csv)
    flat_turns = [t for d in dial for t in d["turns"]]
    if len(flat_turns) != len(res):
        raise ValueError(f"{jsonl} has {len(flat_turns)} messages but {results_csv} has {len(res)} -- "
                         "some classifier requests failed, or results.csv is stale. "
                         "Re-run classify_freerun.py --parse.")
    behaviors = res["pred_behavior"].tolist()

    rows = []
    i = 0
    for d in dial:
        lines = []
        for t in d["turns"]:
            beh = behaviors[i] if pd.notna(behaviors[i]) else "?"
            lines.append(f"[{beh}] {t['role']}: {t['text']}")
            i += 1
        # "deal" is None (no agreement) or the agreed package dict (see
        # two_agents.update_shared_state) -- NOT a bool. Only exists for
        # multi-agent (two_agents.py tracks it via settled_issues/planner
        # decisions) -- single-agent's free-running design has no concept of
        # a deal being reached, so this stays blank for it, same "not
        # applicable to this architecture" pattern as replay's action column.
        deal = d.get("deal")
        rows.append({
            "game_id": d["game_id"],
            "batna_target": d.get("batna_target"),
            "n_messages": len(d["turns"]),
            "deal": ("yes" if deal else "no") if "deal" in d else "",
            "terminal_action": d.get("terminal_action") or "",
            "conversation": "<br>".join(lines),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Assemble
# ---------------------------------------------------------------------------

def main(agent: str, out_dir: str = "data/freerun"):
    run_dir = f"{out_dir}/{agent}"
    results_csv = f"{run_dir}/results.csv"
    jsonl = f"{run_dir}/run.jsonl"
    unlabeled_csv = f"{run_dir}/run_unlabeled.csv"
    out_html = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"{run_dir}/report.html")

    for f in (jsonl, unlabeled_csv, results_csv):
        if not os.path.exists(f):
            sys.exit(f"{f} not found. Run generation/step2_generate.py --agent {agent} --run, then "
                     f"classify/classify_freerun.py --agent {agent} --submit/--check/--parse first.")

    print("loading data + computing the kodis-style divergence table (this takes a minute)...")
    sim = load_sim(results_csv)
    n_dyads = sim.GameId.nunique()
    agent_label = AGENT_LABEL.get(agent, agent)
    kodis_html, kodis_tex = kodis_section(sim, agent_label)
    tex_path = f"{run_dir}/alignment_table.tex"
    with open(tex_path, "w") as fh:
        fh.write(kodis_tex)
    print(f"wrote {tex_path}")
    log_df = build_dialogue_log(jsonl, results_csv)
    log_table_html = log_df.to_html(index=False, na_rep="", escape=False, classes="data")

    deals_note = ""
    if agent == "multi" and (log_df["deal"] == "yes").any() | (log_df["deal"] == "no").any():
        n_deal = int((log_df["deal"] == "yes").sum())
        n_walk = int((log_df["terminal_action"] == "walk_away").sum())
        deals_note = f" — {n_deal}/{n_dyads} reached a deal, {n_walk}/{n_dyads} walked away"

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{agent}-agent free-running report</title>
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
.log-table td {{ max-width: 420px; white-space: normal; word-wrap: break-word; }}
.note {{ font-size: 12px; color: #52514e; margin-top: 8px; }}
</style></head><body>

<h1>{agent}-agent free-running — alignment report</h1>
<div class="subtitle">{agent_label}, {n_dyads} dyads, {len(sim)} model messages{deals_note}
(from {results_csv} / {jsonl})</div>

<h2>Divergence from human negotiators (kodis_stats.py)</h2>
{kodis_html}

<h2>Conversation logs — all {n_dyads} dyads</h2>
<div class="note">One row per DYAD (not per turn -- there's no human counterpart to pair against here,
see the module docstring). Each message is tagged with classify_freerun.py's predicted behavior:
[behavior] role: text. deal/terminal_action are blank for single-agent, which has no concept of a
deal being reached; multi-agent tracks them via two_agents.py's planner decisions.</div>
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
                    help="which step2_generate.py run to report on (freerun/<agent>/).")
    ap.add_argument("--out-dir", default="data/freerun",
                    help="base directory to report on (default: data/freerun, the shipped results).")
    a = ap.parse_args()
    main(a.agent, a.out_dir)
