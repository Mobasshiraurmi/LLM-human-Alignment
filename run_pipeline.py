"""run_pipeline.py — the one entry point to generate every report in this repo.

    python3 run_pipeline.py                     # reports only, from the checked-in results.csv
                                                 # files -- no API key, no cost, about a minute
    python3 run_pipeline.py --full              # regenerate every (architecture, model, design)
                                                 # combo from raw transcripts: generate, classify,
                                                 # then report -- needs API keys, this takes hours.

    python3 run_pipeline.py --full --combo single claude-sonnet-4.6 replay   # just one combo
    python3 run_pipeline.py --full --smoke --combo single gpt-5.4 freerun    # a handful of dyads, cheap
    python3 run_pipeline.py --full --combo single_rag gpt-5.4 replay         # RAG architecture (not
                                                                              # in the default 18 -- see below)

Default mode rebuilds the four kodis-style comparison reports (kodis_reports.py --
model-wise, type-wise × {single, multi}, architecture-wise; see that file's docstring and
kodis_stats.py for the statistics) plus the 18 secondary per-run HTML reports
(report_replay.py / report_freerun.py, one per architecture × model × design combination) --
everything a reviewer needs to see, built entirely from the results.csv files already in
this repo. It then builds the paper's reports into reports/ (report/alignment_reports.py:
Kwon-style mean gaps and Gao-style distribution divergence, by model / type /
architecture, rows from report/results_catalog.json; report/meister_report.py: replay vs
free run) and their figures as vector PDF + SVG in reports/figures/ (PDF export needs
Google Chrome; without it only the SVGs are written).

--full instead rebuilds a combo's data from scratch: step5_replay.py/step2_generate.py --run,
then classify_replay.py/classify_freerun.py (Batch API first, falling back to --sync if the
batch stalls -- the pattern this project settled on after repeatedly hitting that in
practice), then the same report step as above. Every underlying script is resumable on its
own (keyed by what's already written to its output files), so a --full run interrupted
partway through -- including this script getting killed -- can just be re-run.

--full writes into full_run/replay/<key>/ and full_run/freerun/<key>/ by default -- NOT into
data/replay|freerun/, which stays the shipped, already-reviewed results. This way a code
reviewer can run --full for real and compare their own fresh run against the checked-in one,
without any risk of silently overwriting it. Pass --out-dir to pick a different base (e.g.
--out-dir data, if you really do want to regenerate the shipped results in place).

single_rag (generation/single_agent.py, RAG retrieval via stats/rag_align.py) has working
generation code for all three models, but no committed data yet -- it's deliberately left
out of the default 18-combo architecture list, reachable only via --combo single_rag
<model> <design> --full.

Needs OPENAI_API_KEY always (generation on gpt-5.4, and classification for every model),
plus ANTHROPIC_API_KEY for claude-sonnet-4.6 or GEMINI_API_KEY for gemini-2.5-pro combos.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ARCHITECTURES = ["single", "multi", "single_personality"]   # the default 18-combo matrix
MODELS = ["gpt-5.4", "claude-sonnet-4.6", "gemini-2.5-pro"]
DESIGNS = ["replay", "freerun"]

# single_rag has generation code (single_agent.py + rag_align.py's retrieval
# functions) but no committed data yet -- so it's runnable via --combo, but kept
# out of ARCHITECTURES/combos() so the default 18-combo matrix is untouched.
COMBO_ONLY_ARCHITECTURES = ARCHITECTURES + ["single_rag"]

DEFAULT_MODEL = "gpt-5.4"
FULL_RUN_DIR = "full_run"   # --full's default base, kept separate from the shipped data/
CHECK_INTERVAL_S = 30
CHECK_ATTEMPTS = 20          # ~10 minutes of polling before falling back to --sync


def output_key(architecture: str, model: str) -> str:
    return architecture if model == DEFAULT_MODEL else f"{architecture}_{model}"


def combos():
    for architecture in ARCHITECTURES:
        for model in MODELS:
            for design in DESIGNS:
                yield architecture, model, design


def run(cmd: list[str]) -> int:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def report_step(architecture: str, model: str, design: str, out_dir: str = "data") -> None:
    key = output_key(architecture, model)
    script = "report/report_replay.py" if design == "replay" else "report/report_freerun.py"
    rc = run([sys.executable, script, "--agent", key, "--out-dir", f"{out_dir}/{design}"])
    if rc != 0:
        # Warn and keep going rather than sys.exit -- build_reports() loops over 18
        # combos, and one bad report (e.g. from manually-edited data) shouldn't block
        # the other 17.
        print(f"  ! {script} --agent {key} failed (rc={rc}) -- continuing")


def build_reports() -> None:
    print("== kodis-style comparison reports (primary) ==")
    run([sys.executable, "report/kodis_reports.py"])
    print("\n== paper reports: Kwon (mean gaps) + Gao (distributions) by model / type / architecture ==")
    run([sys.executable, "report/alignment_reports.py", "build"])
    print("\n== paper report: replay vs free run (Meister et al. 2025, Eq. 1) ==")
    run([sys.executable, "report/meister_report.py"])
    print("\n== per-run reports (secondary), 18 combos ==")
    for architecture, model, design in combos():
        report_step(architecture, model, design)


def classify_step(architecture: str, model: str, design: str, out_dir: str) -> None:
    key = output_key(architecture, model)
    script = "classify/classify_replay.py" if design == "replay" else "classify/classify_freerun.py"
    combo_dir = f"{out_dir}/{design}"
    rc = run([sys.executable, script, "--agent", key, "--out-dir", combo_dir, "--submit"])
    if rc != 0:
        sys.exit(f"{script} --agent {key} --submit failed (rc={rc})")

    for attempt in range(1, CHECK_ATTEMPTS + 1):
        proc = subprocess.run([sys.executable, script, "--agent", key, "--out-dir", combo_dir, "--check"],
                               capture_output=True, text=True)
        print(proc.stdout, end="")
        # classify_*.py's --check prints "Batch <id>: <status>" on its own line, then
        # (always) a separate "completed=N failed=N total=N" counts line -- that counts
        # line contains the substring "completed" even mid-batch (completed=0), so
        # matching on bare "completed" false-triggers on the very first check. Match the
        # actual status line instead.
        if any(line.strip().endswith(": completed") for line in proc.stdout.splitlines()):
            break
        if attempt < CHECK_ATTEMPTS:
            time.sleep(CHECK_INTERVAL_S)
    else:
        print(f"  batch still not done after {CHECK_ATTEMPTS * CHECK_INTERVAL_S}s -- "
              "falling back to --sync (real-time, ~2x the per-token cost, no queue to stall in)")
        rc = run([sys.executable, script, "--agent", key, "--out-dir", combo_dir, "--sync"])
        if rc != 0:
            sys.exit(f"{script} --agent {key} --sync failed (rc={rc})")
        return

    rc = run([sys.executable, script, "--agent", key, "--out-dir", combo_dir, "--parse"])
    results_csv = f"{combo_dir}/{key}/results.csv"
    if rc != 0 or not os.path.exists(results_csv):
        sys.exit(f"{script} --agent {key} --parse did not produce {results_csv} (rc={rc}) -- "
                 "the batch may not actually be complete; check the --check output above.")


def generate_step(architecture: str, model: str, design: str, out_dir: str, smoke: bool = False) -> None:
    script = "generation/step5_replay.py" if design == "replay" else "generation/step2_generate.py"
    flag = "--smoke" if smoke else "--run"
    combo_dir = f"{out_dir}/{design}"
    rc = run([sys.executable, script, "--agent", architecture, "--model", model, flag, "--out-dir", combo_dir])
    if rc != 0:
        sys.exit(f"{script} --agent {architecture} --model {model} {flag} failed (rc={rc})")


def full_combo(architecture: str, model: str, design: str, out_dir: str, smoke: bool = False) -> None:
    key = output_key(architecture, model)
    print(f"\n=== {key} ({design}){' [smoke]' if smoke else ''} -> {out_dir}/{design}/{key} ===")
    generate_step(architecture, model, design, out_dir, smoke=smoke)
    classify_step(architecture, model, design, out_dir)
    report_step(architecture, model, design, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true",
                     help="regenerate from raw transcripts instead of just rebuilding reports")
    ap.add_argument("--combo", nargs=3, metavar=("ARCHITECTURE", "MODEL", "DESIGN"),
                     help="scope to one (architecture, model, design) combo instead of all 18 "
                          f"-- architecture in {ARCHITECTURES}, model in {MODELS}, design in {DESIGNS}")
    ap.add_argument("--smoke", action="store_true",
                     help="with --full: generate a handful of dyads instead of the full 47 "
                          "(passes --smoke instead of --run to step5_replay.py/step2_generate.py) "
                          "-- for cheaply checking the generate->classify->report chain works at all")
    ap.add_argument("--out-dir", default=FULL_RUN_DIR,
                     help=f"with --full: base directory for the fresh run (default: {FULL_RUN_DIR}/, "
                          "kept separate from data/ so the shipped results are never overwritten). "
                          "Pass --out-dir data to regenerate the shipped results in place instead.")
    a = ap.parse_args()

    if a.combo:
        architecture, model, design = a.combo
        if architecture not in COMBO_ONLY_ARCHITECTURES or model not in MODELS or design not in DESIGNS:
            sys.exit(f"--combo must be one of architecture={COMBO_ONLY_ARCHITECTURES}, "
                     f"model={MODELS}, design={DESIGNS}")
        if a.full:
            full_combo(architecture, model, design, a.out_dir, smoke=a.smoke)
        else:
            report_step(architecture, model, design)
        return

    if a.full:
        for architecture, model, design in combos():
            full_combo(architecture, model, design, a.out_dir, smoke=a.smoke)
        return
    build_reports()


if __name__ == "__main__":
    main()
