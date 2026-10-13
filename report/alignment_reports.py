"""alignment_reports.py — every result, in two report styles x three comparison kinds.

Register each classified results CSV ONCE, tagged with what it is; the reports group
the catalog automatically:

    by model         one table per (architecture, design)   rows = models
    by type          one table per (architecture, model)    rows = replay vs free run
    by architecture  one table per (design, model)          rows = architectures

  Kwon style  stats/kodis_stats.py: group-level gaps (signed, % of baseline, dispersion
              table), dyad-permutation / paired tests -- "do the summaries match?"
  Gao style   stats/gao_style_table.py: Jensen-Shannon divergence between per-dyad
              distributions, dyad permutation -- "does the spread across dyads match?"
              (by-architecture pages also get the Gao Fig. 4 panels)

A table is built when its group has at least 2 rows (a comparison); groups with one row
are listed under the table of contents. Rows whose CSV does not exist yet show "pending".
All tables use the same 47-dyad human baseline (checked automatically).

CATALOG (report/results_catalog.json)
    python3 report/alignment_reports.py list
    python3 report/alignment_reports.py add single-RAG freerun claude-sonnet-4.6 data/freerun/single_rag_claude-sonnet-4.6/results.csv
    python3 report/alignment_reports.py add two-agent freerun gpt-5.4 data/freerun/multi/results.csv --pending
    python3 report/alignment_reports.py remove single-RAG freerun claude-sonnet-4.6
      paths are relative to the repo root
      architecture: single-baseline | single-personality | single-RAG | two-agent
                    (any other name works too; it is shown as given)
      design:       replay | freerun
      model:        gpt-5.4 | claude-sonnet-4.6 | gemini-2.5-pro

BUILD  (-> output/<style>_by_<kind>.html and .tex)
    python3 report/alignment_reports.py build                          # both styles, all three kinds
    python3 report/alignment_reports.py build --style gao --by type --open
    python3 report/alignment_reports.py build --style kwon --by architecture
"""
from __future__ import annotations

import argparse
import contextlib
import html
import io
import json
import os
import sys
import zlib
from typing import NamedTuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))       # report/
ROOT = os.path.dirname(HERE)                            # results paths are relative to the repo root
for _d in ("stats", "generation", "report"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import gao_style_table as G
import kodis_stats as K
import llm_backends as lb

CATALOG = os.path.join(HERE, "results_catalog.json")
HUMAN_CSV = "data/human_data.csv"             # relative to ROOT, as step5_replay.HUMAN_CSV
OUT_DIR = os.path.join(ROOT, "output")
NEEDED = {"GameId", "Sender", "Text", "Phase", "pred_behavior"}

ARCH_ORDER = ["single-baseline", "single-personality", "single-RAG", "two-agent", "debate"]
ARCH_LABEL = {"single-baseline": "Single baseline", "single-personality": "Single + personality",
              "single-RAG": "Single + RAG", "two-agent": "Two-agent", "debate": "Debate agent"}
DESIGNS = ["replay", "freerun"]
DESIGN_LABEL = {"replay": "replay", "freerun": "free run"}
KINDS = {  # kind -> (group-by fields, row field)
    "model": (("architecture", "design"), "model"),
    "type": (("architecture", "model"), "design"),
    "architecture": (("design", "model"), "architecture"),
}


class Entry(NamedTuple):
    architecture: str
    design: str
    model: str
    path: str
    note: str = ""


def arch_label(a: str) -> str:
    return ARCH_LABEL.get(a, a)


def model_label(m: str) -> str:
    return lb.label(m) if m in lb.MODELS else m


def field_label(field: str, value: str) -> str:
    return {"architecture": arch_label, "model": model_label,
            "design": lambda d: DESIGN_LABEL.get(d, d)}[field](value)


def order_key(field: str, value: str):
    seq = {"architecture": ARCH_ORDER, "design": DESIGNS, "model": list(lb.MODELS)}[field]
    return (seq.index(value) if value in seq else len(seq), value)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

def load_catalog() -> list[Entry]:
    if not os.path.exists(CATALOG):
        return []
    return [Entry(**e) for e in json.load(open(CATALOG))]


def save_catalog(entries: list[Entry]):
    entries = sorted(entries, key=lambda e: (order_key("architecture", e.architecture),
                                             order_key("design", e.design), order_key("model", e.model)))
    with open(CATALOG + ".tmp", "w") as f:
        json.dump([{k: getattr(e, k) for k in ("architecture", "design", "model", "path", "note")}
                   for e in entries], f, indent=2)
    os.replace(CATALOG + ".tmp", CATALOG)


def check_csv(path: str):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        sys.exit(f"{path} not found. Fix the path, or add --pending to register it before it exists.")
    cols = set(pd.read_csv(full, nrows=0).columns)
    if NEEDED - cols:
        sys.exit(f"{path}: missing columns {sorted(NEEDED - cols)} -- needs {sorted(NEEDED)}")
    d = pd.read_csv(full, usecols=["GameId"])
    print(f"  {path}: {len(d)} messages, {d.GameId.nunique()} dyads")


def cmd_add(a):
    if a.design not in DESIGNS:
        sys.exit(f"design must be one of {DESIGNS}")
    if not a.pending:
        check_csv(a.csv)
    entries = [e for e in load_catalog() if (e.architecture, e.design, e.model) != (a.architecture, a.design, a.model)]
    replaced = len(entries) != len(load_catalog())
    entries.append(Entry(a.architecture, a.design, a.model, a.csv, a.note or ""))
    save_catalog(entries)
    print(f"{'updated' if replaced else 'added'}: {arch_label(a.architecture)} · {a.design} · {a.model} -> {a.csv}")


def cmd_remove(a):
    entries = load_catalog()
    keep = [e for e in entries if (e.architecture, e.design, e.model) != (a.architecture, a.design, a.model)]
    if len(keep) == len(entries):
        sys.exit("no such entry; see `list`")
    save_catalog(keep)
    print(f"removed: {a.architecture} · {a.design} · {a.model}")


def cmd_list(_a):
    entries = load_catalog()
    print(f"{len(entries)} entries in {os.path.basename(CATALOG)}")
    for e in entries:
        ok = os.path.exists(os.path.join(ROOT, e.path))
        print(f"  [{'ok' if ok else 'pending'}] {e.architecture:19s} {e.design:8s} {e.model:18s} {e.path}")
    for kind in KINDS:
        tables, singles, _ = plan(entries, kind)
        print(f"\nby {kind}: {len(tables)} table(s)" + (f"; single-row groups skipped: {len(singles)}" if singles else ""))
        for title, rows in tables:
            print(f"  - {title}: {', '.join(r[0] for r in rows)}")


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def plan(entries: list[Entry], kind: str):
    """[(table title, [(row label, Entry), ...])], plus the titles of single-row groups."""
    group_fields, row_field = KINDS[kind]
    groups: dict[tuple, list[Entry]] = {}
    for e in entries:
        groups.setdefault(tuple(getattr(e, f) for f in group_fields), []).append(e)
    tables, singles = [], []
    for key in sorted(groups, key=lambda k: tuple(order_key(f, v) for f, v in zip(group_fields, k))):
        rows = sorted(groups[key], key=lambda e: order_key(row_field, getattr(e, row_field)))
        title = " · ".join(field_label(f, v) for f, v in zip(group_fields, key))
        labelled = [(field_label(row_field, getattr(e, row_field)), e) for e in rows]
        (tables if len(rows) >= 2 else singles).append((title, labelled, dict(zip(group_fields, key))))
    return [(t, r) for t, r, _ in tables], [t for t, _, _ in singles], [g for _, _, g in tables]


def load_entry(e: Entry):
    full = os.path.join(ROOT, e.path)
    return K.per_dyad(K.load_model(full)) if os.path.exists(full) else None


def human_baseline(human_csv):
    with contextlib.redirect_stdout(io.StringIO()):
        return K.human_baseline_47(human_csv)


# ---------------------------------------------------------------------------
# Kwon-style section
# ---------------------------------------------------------------------------

CAPTION = {
    "architecture": "Architecture-wise behavioral divergence from human negotiations for {model} in the {design} setting.",
    "model": "Model-wise behavioral divergence from human negotiations for {architecture} in the {design} setting.",
    "type": "Design-wise behavioral divergence from human negotiations for {architecture} on {model}, replay vs. free run.",
}
COMPARED = {"architecture": "architecture conditions", "model": "models", "type": "designs"}


def kwon_section(idx: int, title: str, rows: list, human: dict, cache: dict, kind: str, group: dict):
    """One combined table per comparison (ΔDG + Pre/Post-BATNA shares), as the paper's Table 3."""
    agents, pending, sub, paired, meta = {}, [], {}, {}, []
    for lab, e in rows:
        d = cache.setdefault(e.path, load_entry(e))
        if d is None:
            pending.append(lab)
            sub[lab] = f"{e.design}, results pending"
            continue
        agents[lab] = d
        sub[lab] = f"{field_label('model', e.model)}, {DESIGN_LABEL[e.design]}, {len(d)} dyads"
        paired[lab] = e.design == "replay" and set(d) <= set(human)
        meta.append((lab, e.design, len(d)))
    order = [lab for lab, _ in rows]
    calc = agents or {"__human__": human}
    base_name = next(iter(calc))
    t = K.run_table(K.BEHAVIOR, human, calc, paired)
    kw = dict(order=order, pending=pending, base_name=base_name)
    caption = (f"Table {idx}: " + CAPTION[kind].format(**{f: field_label(f, v) for f, v in group.items()})
               + " Values represent signed gaps relative to the human baseline.")
    notes = K.behavior_notes(meta, COMPARED[kind]) if meta else "<i>Notes.</i> No results yet."
    body = K.html_behavior_table(t, agents, caption, notes, **kw)
    if agents:
        body += (f"<details><summary>Within-group dispersion and underlying numbers</summary>"
                 f"{K.html_dispersion(K.BEHAVIOR, t, agents, f'Table {idx}′: SD across dyads, agent vs human.', sublabels=sub, **kw)}"
                 f"<table class='det'>{K.DETAIL_HEAD}{K.detail_rows(K.BEHAVIOR, t, agents)}</table></details>")
    tex = K.tex_behavior_table(t, agents, caption, notes, **kw)
    baseline = {key: round(t[(key, base_name)]["base"], 10) for *_, key, _, _ in K.BEHAVIOR}
    return body, tex, baseline


# ---------------------------------------------------------------------------
# Gao-style section
# ---------------------------------------------------------------------------

class _FigRow(NamedTuple):
    label: str


def gao_section(idx: int, title: str, rows: list, human: dict, cache: dict, B: int, figures: bool,
                kind: str = "", fig_no: list | None = None):
    """fig_no: one-element running figure counter for the page (figures are numbered page-wide)."""
    hs = G.dyad_scalars(human)
    res, values, sub, agents, pending = {}, {}, {}, [], []
    for lab, e in rows:
        d = cache.setdefault(e.path, load_entry(e))
        if d is None:
            pending.append(lab)
            sub[lab] = f"{e.design}, results pending"
            continue
        m = G.dyad_scalars(d)
        paired = e.design == "replay" and set(m) <= set(hs)
        agents.append(lab)
        sub[lab] = f"{field_label('model', e.model)}, {DESIGN_LABEL[e.design]}, {len(m)} dyads"
        for _, _, key, binning in G.COLUMNS:
            h_vals, m_vals = G.column_values(hs, m, key, paired)
            # seeded by what the row IS, not where its file lives: same row -> same p in every
            # report, and moving or renaming a results file never changes a result
            seed = zlib.crc32(f"{e.architecture}|{e.design}|{e.model}|{key}".encode())
            res[(key, lab)] = G.js_permutation_test(h_vals, m_vals, binning, paired, B=B, seed=seed)
            values[(key, lab)] = m_vals
    order = [lab for lab, _ in rows]
    table = G.html_table(res, agents, order, pending, sub).replace(
        "<div class='cap'>", f"<div class='cap'>Table {idx}: {html.escape(title)}. ", 1)
    figs, fig_tex = "", []
    if figures:
        fig_no = fig_no if fig_no is not None else [1]
        figs, n, fig_tex = G.figures_html(hs, [_FigRow(lab) for lab, _ in rows], None, res, values,
                                          f"by_{kind}_t{idx}", id_prefix=f"t{idx}-", fig_start=fig_no[0],
                                          context=title, row_kind=kind or "architecture",
                                          export_dir=os.path.join(OUT_DIR, "figures"))
        fig_no[0] += n
    return table + figs, "\n\n".join([G.tex_table(res, agents, order, pending)] + fig_tex)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def build(style: str, kind: str, human: dict, human_csv: str, B: int, cache: dict) -> str:
    entries = load_catalog()
    tables, singles, groups = plan(entries, kind)
    by = {"model": "model", "type": "type (replay vs free run)", "architecture": "architecture"}[kind]
    sections, texs, baselines, fig_no = [], [], [], [1]
    for i, (title, rows) in enumerate(tables, start=1):
        if style == "kwon":
            body, tex, base = kwon_section(i, title, rows, human, cache, kind, groups[i - 1])
            baselines.append(base)
        else:
            body, tex = gao_section(i, title, rows, human, cache, B, figures=(kind == "architecture"),
                                    kind=kind, fig_no=fig_no)
        sections.append(f"<h2 id='t{i}'>{i}. {html.escape(title)}</h2>{body}")
        texs.append(f"% ---- {i}. {title} ----\n{tex}")
    if baselines and any(b != baselines[0] for b in baselines):
        sys.exit("human baseline differs between tables -- investigate before using the reports")

    toc = "".join(f"<li><a href='#t{i}'>{html.escape(t)}</a> — {', '.join(html.escape(r[0]) for r in rows)}</li>"
                  for i, (t, rows) in enumerate(tables, start=1))
    skipped = (f"<div class='note'>Groups with a single row (no comparison, not tabled): "
               f"{html.escape('; '.join(singles))}.</div>" if singles else "")
    if style == "kwon":
        intro = K.method_note(len(human), human_csv)
        name = "Kwon/KODIS-style group-level comparison"
    else:
        intro = (f"<div class='note'>Jensen-Shannon divergence between each row's distribution of per-dyad values "
                 f"and the distribution over the {len(human)} human dyads (Gao et al. 2025); {B:,} dyad permutations "
                 "(within-GameId swaps for replay rows, free relabelling for free-run rows); one-sided p. "
                 "Too little dyad-to-dyad variation shows as a large divergence.</div>")
        name = "Gao-style distribution-level comparison"
    css = (K.CSS if style == "kwon" else G.CSS) + "h2 { font-family: -apple-system, sans-serif; font-size: 16px; margin-top: 44px; }"
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{name}, by {by}</title>
<style>{css}</style></head><body>
<h1>{name} — by {by}</h1>
{intro}
<div class="note">{len(tables)} table(s), one per {' × '.join(KINDS[kind][0])}; rows = {KINDS[kind][1]}s. Rows come
from report/results_catalog.json (<code>report/alignment_reports.py add/remove/list</code>).</div>
<ol class="note">{toc}</ol>{skipped}
{''.join(sections)}
</body></html>"""
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{style}_by_{kind}.html")
    open(out, "w").write(page)
    open(out[:-5] + ".tex", "w").write("\n\n".join(texs) + "\n")
    print(f"{style:4s} by {kind:12s} {len(tables)} table(s)"
          + (", baseline identical across tables" if style == "kwon" and len(baselines) > 1 else "")
          + f" -> {os.path.relpath(out, ROOT)}")
    return out


def cmd_build(a):
    styles = ["kwon", "gao"] if a.style == "both" else [a.style]
    kinds = list(KINDS) if a.by == "all" else [a.by]
    human = human_baseline(a.human)
    cache: dict = {}
    for st in styles:
        for kd in kinds:
            path = build(st, kd, human, a.human, a.B, cache)
            if a.open:
                os.system(f'open "{path}"')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("add", help="register (or replace) a results CSV")
    p.add_argument("architecture"); p.add_argument("design"); p.add_argument("model"); p.add_argument("csv")
    p.add_argument("--note"); p.add_argument("--pending", action="store_true")
    p = sp.add_parser("remove", help="drop an entry")
    p.add_argument("architecture"); p.add_argument("design"); p.add_argument("model")
    sp.add_parser("list", help="show the catalog and the tables it produces")
    p = sp.add_parser("build", help="write the reports")
    p.add_argument("--style", default="both", choices=["kwon", "gao", "both"])
    p.add_argument("--by", default="all", choices=["model", "type", "architecture", "all"])
    p.add_argument("--B", type=int, default=G.B_DEFAULT, help="Gao permutations (default 10,000)")
    p.add_argument("--human", default=HUMAN_CSV)
    p.add_argument("--open", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    {"add": cmd_add, "remove": cmd_remove, "list": cmd_list, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
