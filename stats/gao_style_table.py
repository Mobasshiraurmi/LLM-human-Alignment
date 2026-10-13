"""gao_style_table.py — distribution-level divergence report (Gao et al. 2025 style).

A second, independent report alongside stats/kodis_stats.py. That report compares
group-level summaries, which discard dyad-to-dyad variation: a model that reproduces
the human AVERAGE with no variation at all can look well aligned there. This report
never averages. For each per-dyad quantity it compares the full DISTRIBUTION of that
value across the model's dyads with the same distribution across the human dyads, so a
model with the right mean but too little spread produces a LARGE divergence.

Method: Gao, Lee, Burtch & Fazelpour (2025), "Take caution in using LLMs as human
surrogates," PNAS 122(24):e2501660122, SI Appendix section D and Table S5.

PER-DYAD SCALARS (7 columns), from the same results CSVs the KODIS report reads
(per-dyad extraction reused from kodis_stats.per_dyad):
    Pre-BATNA   Neutral / Integrative / Distributive share     in [0, 1]
    Post-BATNA  Neutral / Integrative / Distributive share     in [0, 1]
    DeltaDistributive = post D share - pre D share             in [-1, 1]
(message-length scalars -- words/message, turns/dyad -- were removed from the report)
A dyad with no messages in the relevant phase has no value for that column. For a
replay row (paired: agent dyads share GameIds with human dyads) such a dyad is dropped
from BOTH sides so the two sets stay matched; free-run dyads have no counterpart, so
each side drops its own.

STATISTIC: JS_obs = scipy.spatial.distance.jensenshannon(h_hist, m_hist) ** 2
(natural log, scipy default) between the two histograms of per-dyad values, binned with
FIXED edges computed once per (row, column), before any permutation:
    the six shares        6 equal-width bins over [0, 1]
    DeltaDistributive     6 equal-width bins over [-1, 1]

PERMUTATION TEST (Gao's procedure), B = 10,000, permutation unit = the DYAD:
    replay rows    paired: per GameId, a fair coin decides whether that dyad's human and
                   model values swap groups
    free-run rows  unpaired: pool human + model values, relabel at random preserving the
                   group sizes (exactly Gao's scheme)
    p = (count(JS_null >= JS_obs) + 1) / (B + 1)       one-sided, as Gao
One-sided is correct here: under-dispersion makes JS_obs larger, so nothing hides in the
lower tail. Messages/turns are never permuted (they are nested in dyads; permuting them
is the pseudo-replication that inflated the old pairwise t-test).

Pages: report/alignment_reports.py builds them from report/results_catalog.json:
    python3 report/alignment_reports.py build --style gao              # by model, type, architecture
    python3 report/alignment_reports.py build --style gao --by architecture --open
Writes gao_report_<name>.html and gao_report_<name>.tex. No human baseline row (the
human data is the comparison target, not a row) and no dispersion table (dispersion is
already inside the JSD).
"""
from __future__ import annotations

import argparse
import html
import os
import sys
import zlib

import numpy as np
from scipy.spatial.distance import jensenshannon

HERE = os.path.dirname(os.path.abspath(__file__))       # stats/
ROOT = os.path.dirname(HERE)
for _d in ("stats", "generation", "report"):            # kodis_stats, step5_replay, paper_figures
    sys.path.insert(0, os.path.join(ROOT, _d))
import kodis_stats as k             # per_dyad, load_model + the 47-dyad human baseline

B_DEFAULT = 10_000
N_BINS = 6
SHARE_EDGES = np.linspace(0.0, 1.0, N_BINS + 1)
DELTA_EDGES = np.linspace(-1.0, 1.0, N_BINS + 1)

# (group, label, key, binning) -- binning: "share" | "delta" | "quantile"
COLUMNS = (
    [("Pre-BATNA share", b[0], f"pre_{b}", "share") for b in ("Neutral", "Integrative", "Distributive")]
    + [("Post-BATNA share", b[0], f"post_{b}", "share") for b in ("Neutral", "Integrative", "Distributive")]
    + [("Adaptation", "ΔD", "dD", "delta")]
    # Message-length scalars (words/msg, turns/dyad) were removed from the report; dyad_scalars
    # still computes them and bin_edges still supports "quantile" binning if they return.
)
MODE_TO_REPORTS = {"model": ["models"], "design": ["types"], "architecture": ["architectures"],
                   "all": ["models", "types", "architectures"]}

CAPTION = ("Jensen-Shannon divergence between each architecture's distribution of per-dyad values and "
           "the human distribution over the same quantity. Unlike a comparison of group means, this is "
           "sensitive to how much individual dyads vary: an architecture that reproduces the human average "
           "with little dyad-to-dyad variation yields a LARGE divergence. p-values from 10,000 permutations "
           "of dyad labels (within-GameId swaps for replay rows, free relabelling for free-run rows) under the "
           "null that the observed divergence is due to chance, following Gao et al. (2025). Shares binned "
           "into 6 equal intervals over [0, 1]; ΔDistributive into 6 equal intervals over [-1, 1]. "
           "* p<.05, ** p<.01, *** p<.001. Bold = smallest divergence per column.")


# ---------------------------------------------------------------------------
# Per-dyad scalars
# ---------------------------------------------------------------------------

def dyad_scalars(per: dict) -> dict:
    """{GameId: {column key: value or None}} from kodis_stats.per_dyad output.
    per_dyad stores length as [words/msg / 50, msgs / 12]; undo that scaling here."""
    out = {}
    for g, v in per.items():
        row = {f"{p}_{b}": v[f"{p}_{b}"] for p in ("pre", "post")
               for b in ("Neutral", "Integrative", "Distributive")}
        row["dD"] = v["dD"]
        row["words"] = float(v["length"][0] * 50.0)
        row["turns"] = float(round(v["length"][1] * 12.0))
        out[g] = row
    return out


def column_values(human: dict, model: dict, key: str, paired: bool):
    """(human array, model array). Paired: only GameIds where BOTH sides have a value,
    in the same order, so the swap test exchanges matched dyads."""
    if paired:
        ids = [g for g in human if g in model and human[g][key] is not None and model[g][key] is not None]
        return (np.array([human[g][key] for g in ids], float),
                np.array([model[g][key] for g in ids], float))
    return (np.array([v[key] for v in human.values() if v[key] is not None], float),
            np.array([v[key] for v in model.values() if v[key] is not None], float))


# ---------------------------------------------------------------------------
# Statistic and permutation test
# ---------------------------------------------------------------------------

def bin_edges(binning: str, h: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Fixed edges, computed ONCE from the observed data, never inside the permutation loop."""
    if binning == "share":
        return SHARE_EDGES
    if binning == "delta":
        return DELTA_EDGES
    pooled = np.concatenate([h, m])
    return np.unique(np.quantile(pooled, np.linspace(0, 1, N_BINS + 1)))


def _bin_index(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """np.histogram's convention: half-open bins, the last bin closed on the right."""
    n_bins = len(edges) - 1
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)


def _hist_rows(idx: np.ndarray, n_bins: int) -> np.ndarray:
    """Row-wise normalised histograms of bin indices, shape (rows, values) -> (rows, n_bins)."""
    rows = idx.shape[0]
    flat = idx + np.arange(rows)[:, None] * n_bins
    counts = np.bincount(flat.ravel(), minlength=rows * n_bins).reshape(rows, n_bins)
    return counts / counts.sum(axis=1, keepdims=True)


def _js_rows(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence (natural log) per row == jensenshannon(p, q) ** 2."""
    M = 0.5 * (P + Q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kp = np.where(P > 0, P * np.log(P / M), 0.0).sum(axis=1)
        kq = np.where(Q > 0, Q * np.log(Q / M), 0.0).sum(axis=1)
    return np.maximum(0.5 * kp + 0.5 * kq, 0.0)


def js_permutation_test(h: np.ndarray, m: np.ndarray, binning: str, paired: bool,
                        B: int = B_DEFAULT, seed: int = 0) -> dict:
    edges = bin_edges(binning, h, m)
    n_bins = len(edges) - 1
    base = {"n_h": len(h), "n_m": len(m), "B": B, "edges": edges,
            "scheme": "paired within GameId" if paired else "free relabel"}
    if n_bins < 1 or len(h) == 0 or len(m) == 0:
        return {**base, "js": 0.0, "p": 1.0}
    if n_bins == 1:                     # every value identical: both histograms are [1]
        return {**base, "js": 0.0, "p": 1.0}

    hi, mi = _bin_index(h, edges), _bin_index(m, edges)
    h_hist = np.bincount(hi, minlength=n_bins) / len(h)
    m_hist = np.bincount(mi, minlength=n_bins) / len(m)
    js_obs = float(jensenshannon(h_hist, m_hist) ** 2)
    # the vectorised null must agree with scipy on the observed split
    assert abs(_js_rows(h_hist[None], m_hist[None])[0] - js_obs) < 1e-9

    rng = np.random.default_rng(seed)
    if paired:
        swap = rng.random((B, len(h))) < 0.5
        H = np.where(swap, mi[None, :], hi[None, :])
        M = np.where(swap, hi[None, :], mi[None, :])
    else:
        pooled = np.concatenate([hi, mi])
        order = np.argsort(rng.random((B, len(pooled))), axis=1)
        relabelled = pooled[order]
        H, M = relabelled[:, :len(h)], relabelled[:, len(h):]
    null = _js_rows(_hist_rows(H, n_bins), _hist_rows(M, n_bins))
    p = float((np.sum(null >= js_obs - 1e-12) + 1) / (B + 1))
    return {**base, "js": js_obs, "p": p}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def stars(p: float) -> str:
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def fmt_p(p: float) -> str:
    """Exact p to 4 dp; anything below .0001 (the floor at B=10,000 is 1/10001) as "<.0001"."""
    return "<.0001" if p < 1e-4 else f"{p:.4f}"


def _groups():
    groups = []
    for g, *_ in COLUMNS:
        if not groups or groups[-1][0] != g:
            groups.append([g, 0])
        groups[-1][1] += 1
    return groups


def html_table(res: dict, agents: list, order: list, pending: list, sublabels: dict) -> str:
    top = "".join(f"<th colspan='{n}' class='grp'>{html.escape(g)}</th>" for g, n in _groups())
    head = "".join(f"<th>{html.escape(lab)}</th>" for _, lab, *_ in COLUMNS)
    best = {key: min((res[(key, a)]["js"] for a in agents), default=None) for _, _, key, _ in COLUMNS}
    body = ""
    for name in order:
        tag = f"<br><span class='sub'>{html.escape(sublabels[name])}</span>" if name in sublabels else ""
        if name in pending:
            cells = "".join("<td class='pending'>pending</td>" for _ in COLUMNS)
            body += f"<tr class='pending'><td class='rowh'>{html.escape(name)}{tag}</td>{cells}</tr>"
            continue
        cells = ""
        for _, _, key, _ in COLUMNS:
            r = res[(key, name)]
            v = f"{r['js']:.3f}"
            if len(agents) > 1 and best[key] is not None and np.isclose(r["js"], best[key]):
                v = f"<b>{v}</b>"
            edges = ", ".join(f"{e:.3g}" for e in r["edges"])
            tip = (f"human n={r['n_h']}, model n={r['n_m']}; JS_obs={r['js']:.4f}; p={fmt_p(r['p'])} "
                   f"(exact {r['p']:.6g}); B={r['B']:,}; {r['scheme']}; bin edges [{edges}]")
            cells += f"<td title='{html.escape(tip, quote=True)}'>{v}{stars(r['p'])}</td>"
        body += f"<tr><td class='rowh'>{html.escape(name)}{tag}</td>{cells}</tr>"
    return (f"<table class='kodis'><tr class='r0'><th rowspan='3' class='rowh'>Agent</th>"
            f"<th colspan='{len(COLUMNS)}'>Jensen-Shannon divergence from the human distribution</th></tr>"
            f"<tr class='r1'>{top}</tr><tr class='r2'>{head}</tr>{body}"
            f"<tr class='gaoend'><td colspan='{len(COLUMNS) + 1}'></td></tr></table>"
            f"<div class='cap'>{html.escape(CAPTION)}</div>")


def tex_table(res: dict, agents: list, order: list, pending: list) -> str:
    tex = lambda s: s.replace("Δ", r"$\Delta$").replace("%", r"\%").replace("&", r"\&")
    cols = len(COLUMNS)
    best = {key: min((res[(key, a)]["js"] for a in agents), default=None) for _, _, key, _ in COLUMNS}
    lines = [r"\begin{table*}[t]", r"\centering", r"\begin{tabular}{l" + "c" * cols + "}", r"\toprule",
             rf"\multirow{{3}}{{*}}{{Agent}} & \multicolumn{{{cols}}}{{c}}{{Jensen-Shannon divergence from the human distribution}} \\",
             rf"\cmidrule(lr){{2-{cols + 1}}}",
             " & " + " & ".join(rf"\multicolumn{{{n}}}{{c}}{{{tex(g)}}}" for g, n in _groups()) + r" \\"]
    c, rules = 2, []
    for _, n in _groups():
        rules.append(rf"\cmidrule(lr){{{c}-{c + n - 1}}}")
        c += n
    lines += [" ".join(rules), " & " + " & ".join(tex(lab) for _, lab, *_ in COLUMNS) + r" \\", r"\midrule"]
    for name in order:
        if name in pending:
            lines.append(tex(name) + " & " + " & ".join(["--"] * cols) + r" \\")
            continue
        cells = []
        for _, _, key, _ in COLUMNS:
            r = res[(key, name)]
            v = f"{r['js']:.3f}"
            if len(agents) > 1 and np.isclose(r["js"], best[key]):
                v = rf"\textbf{{{v}}}"
            cells.append(v + (rf"$^{{{stars(r['p'])}}}$" if stars(r["p"]) else ""))
        lines.append(tex(name) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{tex(CAPTION)}}}", r"\end{table*}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure (architecture reports): Gao et al. (2025) Fig. 4 layout, one panel per column
# ---------------------------------------------------------------------------

C_HUMAN, C_AGENT = "#2a78d6", "#eb6834"      # same two series slots as the other reports
C_SIM, C_INK, C_MUTED, C_GRID = "#8f8d85", "#0b0b0b", "#52514e", "#dedcd3"
LN2 = float(np.log(2.0))


def similarity(js: float) -> float:
    """1 - JSD / ln 2: JSD (natural log) is bounded by ln 2, so this is in [0, 1], human = 1."""
    return max(0.0, 1.0 - js / LN2)


def display_edges(binning: str, arrays: list) -> np.ndarray:
    """One common x-axis per panel. Shares and ΔD: exactly the test's fixed edges. Length
    measures: the test uses each row's own pooled-quantile edges, which differ by row, so
    the figure uses 6 equal-width bins over the range of all rows instead."""
    if binning == "share":
        return SHARE_EDGES
    if binning == "delta":
        return DELTA_EDGES
    vals = np.concatenate([a for a in arrays if a is not None and len(a)])
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, N_BINS + 1)


def _props(vals: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.bincount(_bin_index(vals, edges), minlength=len(edges) - 1) / len(vals)


def panel_title(col: tuple) -> str:
    group, lab, key, binning = col
    full = {"N": "Neutral", "I": "Integrative", "D": "Distributive"}
    return (f"{group.split()[0]} {full[lab]} share" if lab in full
            else {"ΔD": "ΔDistributive (post − pre)", "words/msg": "Words per message",
                  "turns/dyad": "Messages (turns) per dyad"}[lab])


def figure_svg(col: tuple, human_vals: np.ndarray, rows: list, sid: str, row_kind: str = "architecture") -> str:
    """One Gao Fig. 4 panel, drawn at its printed size: one ACL column (219 pt), text 6.5-8 pt.
    Human vs model is told apart by FILL (solid vs hatched), not colour alone, so the panel
    reads in grayscale. rows: [(label, values or None if pending, result dict or None)]."""
    import paper_figures as PF
    group, lab, key, binning = col
    title = panel_title(col)
    edges = display_edges(binning, [human_vals] + [v for _, v, _ in rows])
    nb = len(edges) - 1
    all_rows = [("Human", human_vals, None)] + rows
    props = {i: _props(v, edges) for i, (_, v, _) in enumerate(all_rows) if v is not None and len(v)}
    ymax = max(max(p.max() for p in props.values()), 1e-9)

    FS, FS_S = 7.5, 6.5                                   # label / tick sizes in pt
    W = PF.COLUMN_PT
    label_w = min(max(max(len(n) for n, _, _ in all_rows) * 0.5 * FS + 5, 34), 92)
    sim_w, gap = 34, 6
    plot_w = W - label_w - gap - sim_w
    row_h, top = 19, 14
    px0, sx0 = label_w, label_w + plot_w + gap
    X = lambda v: px0 + plot_w * (v - edges[0]) / (edges[-1] - edges[0])
    hatch = f"{sid}-hatch"
    s = [f'<defs><pattern id="{hatch}" patternUnits="userSpaceOnUse" width="3" height="3" '
         f'patternTransform="rotate(45)"><rect width="3" height="3" fill="#ffffff"/>'
         f'<line x1="0" y1="0" x2="0" y2="3" stroke="{C_AGENT}" stroke-width="1.4"/></pattern></defs>',
         f'<text x="0" y="8" font-size="{FS}" font-weight="700" fill="{C_INK}">{html.escape(title)}</text>']

    for i, (name, vals, r) in enumerate(all_rows):
        y0 = top + i * row_h
        inner_top, base = y0 + 2.5, y0 + row_h - 2.5
        Y = lambda prop: base - (base - inner_top) * prop / ymax
        is_h = i == 0
        s.append(f'<rect x="{px0}" y="{y0:.1f}" width="{plot_w:.1f}" height="{row_h - 1}" fill="none" '
                 f'stroke="{C_GRID}" stroke-width="0.5"/>')
        s.append(f'<rect x="{sx0:.1f}" y="{y0:.1f}" width="{sim_w}" height="{row_h - 1}" fill="none" '
                 f'stroke="{C_GRID}" stroke-width="0.5"/>')
        s.append(f'<text x="{label_w - 4:.1f}" y="{y0 + row_h / 2 + 2.5:.1f}" text-anchor="end" font-size="{FS}" '
                 f'font-weight="{700 if is_h else 400}" fill="{C_INK if vals is not None else C_MUTED}">'
                 f'{html.escape(name)}</text>')
        if vals is None or not len(vals):
            s.append(f'<text x="{px0 + plot_w / 2:.1f}" y="{y0 + row_h / 2 + 2.5:.1f}" text-anchor="middle" '
                     f'font-size="{FS_S}" font-style="italic" fill="{C_MUTED}">results pending</text>')
            continue
        pr = props[i]
        fill = C_HUMAN if is_h else f"url(#{hatch})"
        stroke = C_HUMAN if is_h else C_AGENT
        for b in range(nb):
            if pr[b] <= 0:
                continue
            x0, x1 = X(edges[b]) + 0.8, X(edges[b + 1]) - 0.8
            s.append(f'<rect x="{x0:.2f}" y="{Y(pr[b]):.2f}" width="{max(x1 - x0, 0.5):.2f}" '
                     f'height="{base - Y(pr[b]):.2f}" fill="{fill}" stroke="{stroke}" stroke-width="0.6">'
                     f'<title>{html.escape(name)} · {edges[b]:.3g} to {edges[b + 1]:.3g}: '
                     f'{pr[b]:.0%} of {len(vals)} dyads</title></rect>')
        # kernel density (dashed), on the bar scale; omitted when >98% of dyads share one bin (Gao Fig. 4)
        if pr.max() <= 0.98 and len(vals) > 2 and np.std(vals) > 0:
            from scipy.stats import gaussian_kde
            grid = np.linspace(edges[0], edges[-1], 120)
            dens = gaussian_kde(vals)(grid) * (edges[1] - edges[0])
            pts = " ".join(f"{X(g):.2f},{max(Y(min(d, ymax)), inner_top):.2f}" for g, d in zip(grid, dens))
            s.append(f'<polyline points="{pts}" fill="none" stroke="{C_INK}" stroke-width="0.7" '
                     f'stroke-dasharray="2,1.5"/>')
        sim = 1.0 if is_h else similarity(r["js"])
        tip = ("reference distribution" if is_h else
               f"JSD {r['js']:.4f} (natural log), similarity 1 − JSD/ln 2 = {sim:.3f}; p={fmt_p(r['p'])}")
        s.append(f'<rect x="{sx0:.1f}" y="{y0 + 5:.1f}" width="{sim_w * sim:.2f}" height="{row_h - 11}" '
                 f'fill="{C_SIM}"><title>{html.escape(name)}: {html.escape(tip)}</title></rect>')

    yb = top + row_h * len(all_rows)
    labelled = {0, nb // 2, nb} if nb % 2 == 0 else {0, nb}
    for k_, e in enumerate(edges):
        s.append(f'<line x1="{X(e):.2f}" y1="{yb:.1f}" x2="{X(e):.2f}" y2="{yb + 2.5:.1f}" stroke="{C_MUTED}" stroke-width="0.5"/>')
        if k_ in labelled:
            txt = f"{e:.1f}" if binning != "quantile" else f"{e:.0f}"
            anchor = "end" if k_ == nb else "middle"         # keep clear of the similarity axis
            s.append(f'<text x="{X(e):.2f}" y="{yb + 9.5:.1f}" text-anchor="{anchor}" font-size="{FS_S}" '
                     f'fill="{C_INK}">{txt.replace("-", "−")}</text>')
    s.append(f'<text x="{px0 + plot_w / 2:.1f}" y="{yb + 18:.1f}" text-anchor="middle" font-size="{FS_S}" '
             f'fill="{C_INK}">value per dyad</text>')
    for t in (0, 1):
        s.append(f'<line x1="{sx0 + sim_w * t:.2f}" y1="{yb:.1f}" x2="{sx0 + sim_w * t:.2f}" y2="{yb + 2.5:.1f}" '
                 f'stroke="{C_MUTED}" stroke-width="0.5"/>'
                 f'<text x="{sx0 + sim_w * t:.2f}" y="{yb + 9.5:.1f}" text-anchor="{('start', 'end')[t]}" font-size="{FS_S}" '
                 f'fill="{C_INK}">{t}</text>')
    s.append(f'<text x="{sx0 + sim_w / 2:.1f}" y="{yb + 18:.1f}" text-anchor="middle" font-size="{FS_S}" '
             f'fill="{C_INK}">similarity</text>')
    # legend, two lines (fill pattern + line style carry identity; no colour-only keys)
    l1, l2 = yb + 29, yb + 39
    sw = 7
    s.append(f'<rect x="0" y="{l1 - 6}" width="{sw}" height="{sw - 1}" fill="{C_HUMAN}" stroke="{C_HUMAN}" stroke-width="0.6"/>'
             f'<text x="{sw + 3}" y="{l1}" font-size="{FS_S}" fill="{C_INK}">human dyads (solid)</text>'
             f'<rect x="88" y="{l1 - 6}" width="{sw}" height="{sw - 1}" fill="url(#{hatch})" stroke="{C_AGENT}" stroke-width="0.6"/>'
             f'<text x="{88 + sw + 3}" y="{l1}" font-size="{FS_S}" fill="{C_INK}">{html.escape(row_kind)} dyads (hatched)</text>'
             f'<line x1="0" y1="{l2 - 2.5}" x2="{sw + 3}" y2="{l2 - 2.5}" stroke="{C_INK}" stroke-width="0.7" stroke-dasharray="2,1.5"/>'
             f'<text x="{sw + 6}" y="{l2}" font-size="{FS_S}" fill="{C_INK}">kernel density</text>'
             f'<rect x="88" y="{l2 - 6}" width="{sw}" height="{sw - 1}" fill="{C_SIM}"/>'
             f'<text x="{88 + sw + 3}" y="{l2}" font-size="{FS_S}" fill="{C_INK}">similarity = 1 − JSD/ln 2</text>')
    H = l2 + 3
    return PF.svg_root("".join(s), W, H, sid,
                       f"Distribution of {title} per dyad, human dyads vs each {row_kind}")


def figure_caption(col: tuple, context: str, row_kind: str) -> str:
    """Plain-text caption (no 'Figure N:'), shared by the HTML page and the LaTeX snippet."""
    return (f"Per-dyad distribution of the {panel_title(col)}: human dyads (solid bars) and each "
            f"{row_kind}'s dyads (hatched bars), {context}. Bars give the share of dyads per bin and the dashed "
            f"line a kernel density; the bar at right is the similarity to the human distribution, "
            f"1 − JSD/ln 2 (1 = identical; JSD and p in the table).")


def figures_html(human: dict, rows: list, loaded: dict, res: dict, values: dict, name: str,
                 id_prefix: str = "", fig_start: int = 1, context: str = "",
                 row_kind: str = "architecture", export_dir: str | None = None):
    """Returns (html, n_figures, tex snippets). id_prefix keeps SVG ids unique on a page;
    export_dir: also write each panel as <name>_<key>.pdf/.svg at printed size."""
    import paper_figures as PF
    cards, svgs, texs = [], {}, []
    for j, col in enumerate(COLUMNS):
        key = col[2]
        human_vals = column_values(human, human, key, paired=False)[0]
        fig_rows = [(r.label, values.get((key, r.label)), res.get((key, r.label))) for r in rows]
        sid = f"fig-{id_prefix}{j}"
        fname = f"gao_{name}_{key}"
        svg = figure_svg(col, human_vals, fig_rows, sid, row_kind)
        cap = figure_caption(col, context, row_kind)
        svgs[fname] = svg
        buttons = (f"<div class='downloads'><button onclick=\"downloadChart('{sid}', '{fname}', 'png')\">"
                   f"Download PNG</button><button onclick=\"downloadChart('{sid}', '{fname}', 'svg')\">"
                   f"Download SVG</button>" + (f"<span class='pdfnote'>PDF: figures/{fname}.pdf</span>" if export_dir else "")
                   + "</div>")
        cards.append(f"<div class='chart-card'>{PF.figure_html(svg, fig_start + j, html.escape(cap), buttons)}</div>")
        texs.append(PF.tex_figure(f"figures/{fname}.pdf", PF.tex_text(cap), wide=False, label=f"fig:{fname}"))
    if export_dir:
        PF.export(svgs, export_dir)
    page = (f"<script>{PF.DOWNLOAD_SCRIPT}</script><h3 class='fighead'>Per-dyad distributions (Gao et al. 2025, Fig. 4 layout)</h3>"
            "<div class='note'>One figure per column of the table, drawn at the printed width of one ACL column "
            "(219 pt) with 6.5-8 pt text; the PDF next to the report is the vector version for the paper. Hover a bar "
            "for its numbers. Density curves are omitted where more than 98% of dyads fall in one bin; every panel uses "
            "the test's own fixed bins. Human row: all dyads with a value; replay rows are tested on GameIds present on "
            "both sides.</div>"
            f"<div class='figs'>{''.join(cards)}</div>")
    return page, len(COLUMNS), texs


import paper_figures as _PF
CSS = k.CSS + _PF.CSS + """
table.kodis tr.gaoend td { border-top: 1.5px solid #000; padding: 0; }
.figs { display: grid; grid-template-columns: repeat(auto-fill, minmax(480px, 1fr)); gap: 18px; max-width: 1600px; }
.pdfnote { font-family: -apple-system, sans-serif; font-size: 11px; color: #6b6a65; align-self: center; }
h3.fighead { font-family: -apple-system, sans-serif; font-size: 15px; margin-top: 30px; }
.chart-card { border: 1px solid #dedcd3; border-radius: 8px; padding: 12px; background: #fff; }
.downloads { display: flex; gap: 8px; margin-top: 6px; }
.downloads button { font: inherit; font-size: 12px; padding: 4px 10px; border: 1px solid #dedcd3;
  border-radius: 6px; background: #fcfcfb; cursor: pointer; }
"""


# ---------------------------------------------------------------------------
# Pages are built by report/alignment_reports.py (by model / type / architecture, from
# report/results_catalog.json); this module is the statistics + figure engine.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit("gao_style_table.py is a library; build the reports with\n"
             "    python3 report/alignment_reports.py build --style gao")
