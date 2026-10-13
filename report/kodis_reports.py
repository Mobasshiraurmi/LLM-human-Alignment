"""kodis_reports.py — the KODIS-style comparison reports, rows kept in report_registry.json.
Ported/rebuilt from Urmi_assmebledall's kodis_reports.py against this repo's own 18
(architecture × model × design) results.csv files, using kodis_stats.py's corrected
permutation-of-dyads engine (see that file's docstring for the statistics).

BUILD
    python3 report/kodis_reports.py                          # every report
    python3 report/kodis_reports.py --report models          # or: types_single, types_multi, architectures
    python3 report/kodis_reports.py --open

ADD / CHANGE ROWS FROM THE COMMAND LINE (saved to report_registry.json, report rebuilt)
    python3 report/kodis_reports.py --list
    python3 report/kodis_reports.py --set models "Single (GPT-4.1)" replay/single_gpt-4.1/results.csv
    python3 report/kodis_reports.py --remove architectures "Two-agent (GPT-5.4)"
    python3 report/kodis_reports.py --new-report extra --title "..." --who "..."

--set adds the row, or updates it when the label already exists (keeping its position).
The CSV must exist and have GameId, Sender, Text, Phase, pred_behavior (any classifier
output does); pass --pending to register a path that will exist later. --design is
replay|freerun; for a new row it defaults to freerun when the path says "freerun",
otherwise replay. Labels are matched exactly, so quote them.

Writes kodis_report_<name>.html and kodis_report_<name>.tex. Rows whose results CSV does
not exist yet are shown as "pending". One fixed human baseline (the same 47 BATNA-identified
dyads as every other report in this repo) is used for every report and every row, so the
numbers are comparable across reports.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from typing import NamedTuple

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repo root -- where report_registry.json and every results.csv live
sys.path.insert(0, os.path.join(ROOT, "stats"))        # kodis_stats
sys.path.insert(0, os.path.join(ROOT, "generation"))   # step5_replay
import kodis_stats as k
import step5_replay as sr

NEEDED = {"GameId", "Sender", "Text", "Phase", "pred_behavior"}
REGISTRY = os.path.join(HERE, "report_registry.json")
DEFAULT_HUMAN_CSV = sr.HUMAN_CSV


class Row(NamedTuple):
    label: str
    design: str        # "replay" | "freerun"
    results: str       # classified results CSV, relative to this folder
    note: str = ""


# ---------------------------------------------------------------------------
# Registry (report_registry.json): {report: {"title", "who", "rows": [Row, ...]}}
# ---------------------------------------------------------------------------

def load_registry() -> dict:
    with open(REGISTRY) as f:
        data = json.load(f)
    for spec in data.values():
        spec["rows"] = [Row(**r) for r in spec["rows"]]
    return data


def save_registry(data: dict):
    out = {n: {**spec, "rows": [r._asdict() for r in spec["rows"]]} for n, spec in data.items()}
    with open(REGISTRY + ".tmp", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    os.replace(REGISTRY + ".tmp", REGISTRY)


def check_csv(path: str):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        sys.exit(f"{path} not found (relative to {ROOT}). Fix the path, or add --pending "
                 "to register it before the file exists.")
    cols = set(pd.read_csv(full, nrows=0).columns)
    if NEEDED - cols:
        sys.exit(f"{path}: missing columns {sorted(NEEDED - cols)} -- needs {sorted(NEEDED)}")
    d = pd.read_csv(full, usecols=["GameId", "pred_behavior"])
    print(f"  {path}: {len(d)} messages, {d.GameId.nunique()} dyads"
          + (f", {d.pred_behavior.isna().sum()} unlabelled" if d.pred_behavior.isna().any() else ""))


def set_row(data, report, label, path, design=None, note=None, pending=False):
    if report not in data:
        sys.exit(f"no report {report!r}; reports: {list(data)} (or create one with --new-report)")
    if not pending:
        check_csv(path)
    rows = data[report]["rows"]
    i = next((j for j, r in enumerate(rows) if r.label == label), None)
    if i is None:
        design = design or ("freerun" if "freerun" in path.lower() else "replay")
        rows.append(Row(label, design, path, note or ""))
        print(f"added to {report}: {label!r} ({design}) -> {path}")
    else:
        old = rows[i]
        rows[i] = Row(label, design or old.design, path, old.note if note is None else note)
        print(f"updated in {report}: {label!r} ({rows[i].design}) {old.results} -> {path}")


def list_registry(data):
    for name, spec in data.items():
        print(f"\n{name}: {spec['title']}")
        for r in spec["rows"]:
            ok = os.path.exists(os.path.join(ROOT, r.results))
            print(f"  [{'ok' if ok else 'pending'}] {r.label!r:45s} {r.design:8s} {r.results}")


def load_row(row):
    """Per-dyad data for a row, or None when its results CSV is not there yet."""
    path = os.path.join(ROOT, row.results)
    if not os.path.exists(path):
        return None
    cols = set(pd.read_csv(path, nrows=0).columns)
    missing = NEEDED - cols
    if missing:
        sys.exit(f"{row.results}: missing columns {sorted(missing)} -- needs {sorted(NEEDED)}")
    return k.per_dyad(k.load_model(path))


def build(name, spec, human, human_csv):
    rows = spec["rows"]
    loaded = {r.label: load_row(r) for r in rows}
    agents = {lab: d for lab, d in loaded.items() if d is not None}
    pending = [lab for lab, d in loaded.items() if d is None]
    order = [r.label for r in rows]
    sub = {r.label: (f"{r.design}, {len(loaded[r.label])} dyads" if loaded[r.label] is not None
                     else f"{r.design}, results pending") for r in rows}

    # Replay rows answer the real dyads (same GameIds) -> tested paired. A replay CSV whose
    # GameIds are not all among the human dyads is tested unpaired, with a warning.
    paired = {}
    for r in rows:
        d = loaded[r.label]
        if d is None:
            continue
        paired[r.label] = r.design == "replay" and set(d) <= set(human)
        if r.design == "replay" and not paired[r.label]:
            print(f"  ! {r.label}: replay CSV has GameIds outside the 47 human dyads -- tested unpaired")
    # With no results yet, run the baseline against itself only to fill the human row.
    calc = agents or {"__human__": human}
    base_name = next(iter(calc))
    t1, t2 = (k.run_table(k.TABLE1, human, calc, paired),
              k.run_table(k.TABLE2, human, calc, paired))
    cap1 = (f"Table 1: Divergence metric results for {spec['who']} compared to human negotiators, "
            "along with the human baseline.")
    cap2 = (f"Table 2: Behavior-share gaps by phase for {spec['who']} compared to human negotiators, "
            "along with the human baseline.")
    kw = dict(order=order, pending=pending, base_name=base_name)

    notes = "".join(
        f"<tr><td>{html.escape(r.label)}</td><td>{r.design}</td><td>{html.escape(r.results)}</td>"
        f"<td>{'<b>pending</b>' if r.label in pending else 'loaded'}</td><td>{html.escape(r.note)}</td></tr>"
        for r in rows)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(spec['title'])}</title>
<style>{k.CSS}</style></head><body>
<h1>{html.escape(spec['title'])}</h1>
{k.method_note(len(human), human_csv)}
<div class="note">{len(agents)} of {len(rows)} rows have results; pending rows fill in once their CSV exists
(report_registry.json, or <code>kodis_reports.py --set</code>).</div>
{k.html_table(k.TABLE1, t1, agents, cap1, len(human), sublabels=sub, **kw)}
{k.html_dispersion(k.TABLE1, t1, agents, "Table 1b: Within-group dispersion for the metrics of Table 1.", sublabels=sub, **kw) if agents else ''}
{k.html_table(k.TABLE2, t2, agents, cap2, len(human), sublabels=sub, **kw)}
{k.html_dispersion(k.TABLE2, t2, agents, "Table 2b: Within-group dispersion for the metrics of Table 2.", sublabels=sub, **kw) if agents else ''}
<h1>Rows</h1>
<table class="det"><tr><th>row</th><th>design</th><th>results CSV</th><th>status</th><th>note</th></tr>
{notes}</table>
<h1>Underlying numbers</h1>
<table class="det">{k.DETAIL_HEAD}
{k.detail_rows(k.TABLE1 + k.TABLE2, {**t1, **t2}, agents) if agents else ''}</table>
</body></html>"""
    out_html = os.path.join(HERE, f"kodis_report_{name}.html")
    with open(out_html, "w") as f:
        f.write(page)
    with open(os.path.join(HERE, f"kodis_report_{name}.tex"), "w") as f:
        f.write(k.tex_table(k.TABLE1, t1, agents, cap1, len(human), **kw) + "\n\n"
                + k.tex_table(k.TABLE2, t2, agents, cap2, len(human), **kw) + "\n")
    print(f"{name:14s} {len(agents)}/{len(rows)} rows with results"
          + (f"   pending: {', '.join(pending)}" if pending else "") + f"\n  -> {out_html}")
    baseline = {key: round(t[(key, base_name)]["base"], 10)
                for spec_, t in ((k.TABLE1, t1), (k.TABLE2, t2)) for *_, key, _, _ in spec_}
    return out_html, baseline


def build_reports(names, human_csv, open_after=False):
    """Build the named reports and check the human baseline row is identical in all of them
    (they share one 47-dyad baseline, so any difference means a computation went wrong)."""
    data = load_registry()
    unknown = [n for n in names if n not in data]
    if unknown:
        sys.exit(f"no report {unknown}; reports: {list(data)}")
    human = k.human_baseline_47(human_csv)
    if len(human) != 47:
        print(f"  ! human baseline has {len(human)} dyads, expected 47")
    baselines = {}
    for n in names:
        if not data[n]["rows"]:
            print(f"{n}: no rows yet -- add some with --set")
            continue
        path, baselines[n] = build(n, data[n], human, human_csv)
        if open_after:
            os.system(f'open "{path}"')
    if len(baselines) > 1:
        first = next(iter(baselines.values()))
        same = all(b == first for b in baselines.values())
        print(f"baseline check: human baseline row {'IDENTICAL' if same else 'DIFFERS'} across "
              f"{', '.join(baselines)} ({len(human)} dyads)")
        if not same:
            sys.exit("baseline rows differ between reports -- investigate before using the tables")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default="all", help="report to build (default: all)")
    ap.add_argument("--human", default=DEFAULT_HUMAN_CSV)
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--list", action="store_true", help="show every report's rows and status")
    ap.add_argument("--set", nargs=3, metavar=("REPORT", "LABEL", "CSV"),
                    help="add a row, or update the row with this label")
    ap.add_argument("--remove", nargs=2, metavar=("REPORT", "LABEL"), help="delete a row")
    ap.add_argument("--design", choices=["replay", "freerun"], help="with --set")
    ap.add_argument("--note", help="with --set: note shown in the report's Rows table")
    ap.add_argument("--pending", action="store_true", help="with --set: allow a CSV that does not exist yet")
    ap.add_argument("--new-report", metavar="NAME", help="create an empty report (then add rows with --set)")
    ap.add_argument("--title", help="with --new-report")
    ap.add_argument("--who", help="with --new-report: completes 'Table 1: Divergence metric results for ...'")
    a = ap.parse_args()
    os.chdir(ROOT)
    data = load_registry()

    changed = None
    if a.new_report:
        if a.new_report in data:
            sys.exit(f"report {a.new_report!r} already exists")
        data[a.new_report] = {"title": a.title or a.new_report, "who": a.who or a.new_report, "rows": []}
        changed = a.new_report
        print(f"created report {a.new_report!r}")
    if a.set:
        report, label, path = a.set
        set_row(data, report, label, path, a.design, a.note, a.pending)
        changed = report
    if a.remove:
        report, label = a.remove
        rows = data.get(report, {}).get("rows", [])
        if not any(r.label == label for r in rows):
            sys.exit(f"no row {label!r} in {report!r}; see --list")
        data[report]["rows"] = [r for r in rows if r.label != label]
        changed = report
        print(f"removed from {report}: {label!r}")
    if changed:
        save_registry(data)
    if a.list:
        list_registry(data)
        if not changed:
            return

    names = [changed] if changed else (list(data) if a.report == "all" else [a.report])
    build_reports(names, a.human, open_after=a.open)


if __name__ == "__main__":
    main()
