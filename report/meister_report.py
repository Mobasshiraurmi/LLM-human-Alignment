"""meister_report.py — replay vs free run: which design is closer to the human dyads?
Meister et al. (NAACL 2025), Eq. 1, adapted.

UNIT. The dyad plays the role of Meister's question; the phase (Pre / Post-BATNA) plays
the role of their demographic group; the three behaviour categories (Neutral,
Integrative, Distributive) are the answer options.

PER-DYAD DISTANCE, for human dyad i and phase g:
    y[g,i]            the human dyad's empirical mix in phase g (shares of its messages)
    yhat_replay[g,i]  the replay of the same GameId
    d_replay[g,i]   = JSD(y[g,i], yhat_replay[g,i])
    d_free[g,i]     = mean over free-run dyads j of JSD(y[g,i], yhat_free[g,j])
JSD = scipy.spatial.distance.jensenshannon(p, q) ** 2 (natural log, <= ln 2), the
convention of strategic_behavior.py.

FREE-RUN PAIRING (decision recorded here). Free-run dyads are synthetic and mirror no
human GameId, so the "mirroring free-run dyad" of the method does not exist. d_free[g,i]
is therefore the POPULATION AVERAGE: the expected distance from human dyad i to a dyad of
that free run. Pairing stays per human dyad i. Note the asymmetry this cannot remove: a
replay dyad answers dyad i's actual conversation, a free-run dyad never sees it.

AGGREGATE (Eq. 1). A(design) = (1/|G|) sum_g [ mean_i d[g,i] ], G = {Pre, Post}: equal
weight per phase, mean over dyads within a phase. Lower = closer to humans.

ROWS
    Overall mix       JSD, all-phase mix
    Pre-BATNA mix     JSD
    Post-BATNA mix    JSD
    Eq. 1 aggregate   mean of the Pre and Post rows (equal phase weight)
    ΔDistributive     ABSOLUTE ERROR |dD_human - dD_model| (scalar -- not a JSD); only dyads
                      reaching both phases, so its n is smaller
    (breakdown)       six shares, absolute error -- components of the mix rows, not
                      independent findings

TESTS. Statistic: Delta = A(free run) - A(replay); > 0 means replay is closer.
  Primary    paired bootstrap: resample the human dyad ids with replacement (the SAME ids for
             both designs -- this keeps the pairing) and, because d_free averages over a
             finite free-run sample, also resample the free-run dyads; recompute both A's;
             record Delta. 1,000 resamples reported; a 10,000-resample run checks stability.
             95% percentile CI; a CI excluding 0 is the inference.
  Secondary  Wilcoxon signed-rank, two-sided, on the per-dyad differences d_free - d_replay
             (per phase and on the phase-averaged per-dyad value). Non-parametric; assumes
             the differences are roughly symmetric about their median.
  Floor      sign test (exact binomial on the signs) -- drops even that assumption.
  The secondary tests treat d_free[i] as fixed (they cannot see free-run sampling
  noise); the bootstrap includes it. No t-test anywhere.

    python3 report/meister_report.py --open             # every (architecture, model) with both designs
    python3 report/meister_report.py --B 1000 --check-B 10000
Reads report/results_catalog.json (alignment_reports.py add/list); writes
output/meister_replay_vs_freerun.html and .tex.
"""
from __future__ import annotations

import argparse
import contextlib
import html
import io
import os
import sys

import warnings
import zlib

import numpy as np
from scipy.spatial.distance import jensenshannon

# dyads with no value for a metric (e.g. no Post-BATNA phase) are NaN by design
warnings.filterwarnings("ignore", message="Mean of empty slice")
from scipy.stats import binomtest, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))       # report/
ROOT = os.path.dirname(HERE)
for _d in ("stats", "generation", "report"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import alignment_reports as AR
import kodis_stats as K

B_MAIN, B_CHECK, SEED = 1000, 10000, 20261007
CATS = ("Neutral", "Integrative", "Distributive")
# (row label, kind, per_dyad key, distance label)
ROWS = [("Overall mix", "mix", "all", "JSD"),
        ("Pre-BATNA mix", "mix", "pre", "JSD"),
        ("Post-BATNA mix", "mix", "post", "JSD"),
        ("Eq. 1 aggregate (Pre, Post)", "eq1", None, "JSD, mean of phases"),
        ("ΔDistributive", "abs", "dD", "absolute error")]
BREAKDOWN = [(f"{ph} {c}", "abs", f"{k}_{c}", "absolute error")
             for k, ph in (("pre", "Pre-BATNA"), ("post", "Post-BATNA")) for c in CATS]


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

def _js_matrix(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """JSD (natural log) between every row of P and every row of Q == jensenshannon(p, q) ** 2."""
    p, q = P[:, None, :], Q[None, :, :]
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kp = np.where(p > 0, p * np.log(p / m), 0.0).sum(-1)
        kq = np.where(q > 0, q * np.log(q / m), 0.0).sum(-1)
    return np.maximum(0.5 * kp + 0.5 * kq, 0.0)


def metric_arrays(human: dict, replay: dict, free: dict, kind: str, key: str):
    """Per human dyad (fixed order `ids`): d_replay[i], and the free-run distance matrix
    D[i, j] (human i vs free dyad j). NaN where a dyad has no value for this metric."""
    ids = sorted(human)
    fids = sorted(free)
    hv = [human[g][key] for g in ids]
    rv = [replay.get(g, {}).get(key) if g in replay else None for g in ids]
    fv = [free[j][key] for j in fids]
    n, m = len(ids), len(fids)
    d_rep = np.full(n, np.nan)
    D = np.full((n, m), np.nan)
    if kind == "mix":
        hm = [i for i in range(n) if hv[i] is not None]
        fm = [j for j in range(m) if fv[j] is not None]
        for i in hm:
            if rv[i] is not None:
                d_rep[i] = jensenshannon(hv[i], rv[i]) ** 2
        if hm and fm:
            sub = _js_matrix(np.array([hv[i] for i in hm]), np.array([fv[j] for j in fm]))
            D[np.ix_(hm, fm)] = sub
    else:
        for i in range(n):
            if hv[i] is not None and rv[i] is not None:
                d_rep[i] = abs(hv[i] - rv[i])
            if hv[i] is not None:
                for j in range(m):
                    if fv[j] is not None:
                        D[i, j] = abs(hv[i] - fv[j])
    # a human dyad counts for this metric only if it has a replay value AND some free value
    valid = ~np.isnan(d_rep) & ~np.all(np.isnan(D), axis=1)
    d_rep[~valid] = np.nan
    D[~valid] = np.nan
    return ids, d_rep, D


# ---------------------------------------------------------------------------
# Bootstrap (paired over human dyads, plus resampled free-run dyads)
# ---------------------------------------------------------------------------

def _boot_means(d_rep, D, Ci, Cj):
    """Per replicate: A_replay, A_free for one metric. Ci (B x n human counts), Cj (B x m free counts)."""
    vmask = ~np.isnan(d_rep)
    Dz = np.nan_to_num(D)
    have = (~np.isnan(D)).astype(float)
    num = Cj @ Dz.T                       # (B, n): sum over resampled free dyads of D[i, j]
    den = Cj @ have.T                     # (B, n): how many of them had a value
    with np.errstate(invalid="ignore", divide="ignore"):
        dfree = num / den                 # mean over resampled free dyads (per replicate)
    w = Ci * vmask[None, :] * (den > 0)
    a_rep = (w * np.nan_to_num(d_rep)[None, :]).sum(1) / w.sum(1)
    a_free = (w * np.nan_to_num(dfree)).sum(1) / w.sum(1)
    return a_rep, a_free


def bootstrap(metrics: dict, n: int, m: int, B: int, seed: int) -> dict:
    """metrics: {key: (d_rep, D)}. One shared draw of human ids and free dyads per replicate,
    so every metric (and the Eq. 1 aggregate) is computed on the same resample."""
    rng = np.random.default_rng(seed)
    Ci = rng.multinomial(n, np.full(n, 1 / n), size=B).astype(float)
    Cj = rng.multinomial(m, np.full(m, 1 / m), size=B).astype(float)
    return {k: _boot_means(dr, D, Ci, Cj) for k, (dr, D) in metrics.items()}


def ci(x):
    return float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))


# ---------------------------------------------------------------------------
# One comparison (an architecture x model with both designs)
# ---------------------------------------------------------------------------

def analyse(human, replay, free, B, B_check, seed):
    ids_all = sorted(human)
    metrics, per_dyad_diff = {}, {}
    for label, kind, key, _ in ROWS + BREAKDOWN:
        if kind == "eq1":
            continue
        ids, d_rep, D = metric_arrays(human, replay, free, kind, key)
        metrics[key] = (d_rep, D)
        per_dyad_diff[key] = np.nanmean(D, axis=1) - d_rep          # d_free - d_replay per human dyad
    n, m = len(ids_all), len(free)

    def summarise(Bn, sd):
        boot = bootstrap(metrics, n, m, Bn, sd)
        out = {}
        for label, kind, key, _ in ROWS + BREAKDOWN:
            if kind == "eq1":
                ar = 0.5 * (boot["pre"][0] + boot["post"][0])
                af = 0.5 * (boot["pre"][1] + boot["post"][1])
            else:
                ar, af = boot[key]
            out[label] = {"rep_ci": ci(ar), "free_ci": ci(af), "delta_ci": ci(af - ar)}
        return out

    main, check = summarise(B, seed), summarise(B_check, seed + 1)
    res = {}
    for label, kind, key, dist in ROWS + BREAKDOWN:
        if kind == "eq1":
            d_rep = 0.5 * (metrics["pre"][0] + metrics["post"][0])     # dyads with both phases
            d_free = 0.5 * (np.nanmean(metrics["pre"][1], 1) + np.nanmean(metrics["post"][1], 1))
            a_rep = 0.5 * (np.nanmean(metrics["pre"][0]) + np.nanmean(metrics["post"][0]))
            a_free = 0.5 * (np.nanmean(np.nanmean(metrics["pre"][1], 1)) +
                            np.nanmean(np.nanmean(metrics["post"][1], 1)))
            diff = d_free - d_rep
            n_used = f"{int(np.sum(~np.isnan(metrics['pre'][0])))} pre / {int(np.sum(~np.isnan(metrics['post'][0])))} post"
        else:
            d_rep, D = metrics[key]
            a_rep, a_free = float(np.nanmean(d_rep)), float(np.nanmean(np.nanmean(D, 1)))
            diff = per_dyad_diff[key]
            n_used = str(int(np.sum(~np.isnan(d_rep))))
        diff = diff[~np.isnan(diff)]
        try:
            p_w = float(wilcoxon(diff, alternative="two-sided").pvalue) if np.any(diff != 0) else 1.0
        except ValueError:
            p_w = 1.0
        nz = diff[diff != 0]
        p_s = float(binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue) if len(nz) else 1.0
        mc, cc = main[label]["delta_ci"], check[label]["delta_ci"]
        res[label] = {
            "dist": dist, "n": n_used, "n_pairs_test": len(diff),
            "rep": a_rep, "free": a_free, "delta": a_free - a_rep, **main[label],
            "delta_ci_check": cc, "ci_shift": max(abs(mc[0] - cc[0]), abs(mc[1] - cc[1])),
            "excl0": mc[0] > 0 or mc[1] < 0, "excl0_check": cc[0] > 0 or cc[1] < 0,
            "p_wilcoxon": p_w, "p_sign": p_s,
        }
    return res


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def fp(p):
    return "<.001" if p < 0.001 else f"{p:.3f}"


def fci(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def row_html(label, r, sub=False):
    d = f"{r['delta']:+.3f}".replace("-", "−")
    if r["excl0"]:
        d = f"<b>{d}</b>"
    cls = " class='sub'" if sub else ""
    return (f"<tr{cls}><td class='rowh'>{html.escape(label)}</td><td>{r['dist']}</td><td>{r['n']}</td>"
            f"<td>{r['rep']:.3f} <span class='ci'>{fci(r['rep_ci'])}</span></td>"
            f"<td>{r['free']:.3f} <span class='ci'>{fci(r['free_ci'])}</span></td>"
            f"<td>{d} <span class='ci'>{fci(r['delta_ci'])}</span></td>"
            f"<td>{fp(r['p_wilcoxon'])}</td><td>{fp(r['p_sign'])}</td></tr>")


def section(idx, title, res, n_free, B, B_check):
    moved = max(r["ci_shift"] for r in res.values())
    flips = [lab for lab, r in res.items() if r["excl0"] != r["excl0_check"]]
    stability = (f"Stability: with {B_check:,} resamples the Δ CIs moved by at most {moved:.4f}"
                 + ("; no row changed whether its CI excludes 0." if not flips else
                    f"; the excludes-0 decision CHANGED for: {', '.join(flips)}."))
    head = ("<tr><th class='rowh'>Metric</th><th>distance</th><th>n dyads</th>"
            "<th>Replay mean [95% CI]</th><th>Free run mean [95% CI]</th>"
            "<th>Δ = free − replay [95% CI]</th><th>Wilcoxon p</th><th>sign-test p</th></tr>")
    main_rows = "".join(row_html(lab, res[lab]) for lab, *_ in ROWS)
    brk = "".join(row_html(lab, res[lab], sub=True) for lab, *_ in BREAKDOWN)
    cap = (f"Table {idx + 1}: Distance to the human dyads, replay vs free run — {html.escape(title)} "
           "(Meister et al. 2025, Eq. 1). Lower = closer to humans; Δ > 0 = replay closer.")
    notes = (f"<i>Notes.</i> Per human dyad, replay distance = JSD (natural log) between the human dyad's behaviour mix and "
             f"its replay; free-run distance = mean JSD to all {n_free} free-run dyads (free runs mirror no human dyad). "
             "Eq. 1 aggregate = mean of the Pre- and Post-BATNA rows, equal phase weight. ΔDistributive is an absolute "
             "error (|dD<sub>human</sub> − dD<sub>model</sub>|), not a JSD, over dyads reaching both phases. "
             f"<b>Primary inference:</b> 95% percentile CI on Δ from a paired bootstrap ({B:,} resamples of the human dyad "
             "ids, shared by both designs, with the free-run dyads resampled too); bold Δ = CI excludes 0. "
             "<b>Secondary:</b> two-sided Wilcoxon signed-rank on the per-dyad differences. <b>Assumption-free floor:</b> "
             "sign test. Both treat each dyad's free-run average as fixed; the bootstrap does not. No t-test is used. "
             f"{stability}")
    return (f"<h2 id='t{idx}'>{idx}. {html.escape(title)}</h2>"
            f"<table class='kodis mt'>{head}{main_rows}</table><div class='tcap'>{cap}</div>"
            f"<details><summary>Breakdown: the six shares (components of the mix rows -- not independent findings)</summary>"
            f"<table class='kodis mt'>{head}{brk}</table></details>"
            f"<div class='foot'>{notes}</div>")


def tex_section(title, res, n_free, B):
    t = lambda s: s.replace("Δ", r"$\Delta$").replace("−", "$-$").replace("&", r"\&")
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\begin{tabular}{lllccccc}", r"\toprule",
             r"Metric & distance & $n$ & Replay [95\% CI] & Free run [95\% CI] & $\Delta$ [95\% CI] & Wilcoxon $p$ & sign $p$ \\",
             r"\midrule"]
    for lab, *_ in ROWS:
        r = res[lab]
        d = f"{r['delta']:+.3f}".replace("-", "$-$")
        if r["excl0"]:
            d = rf"\textbf{{{d}}}"
        lines.append(f"{t(lab)} & {r['dist']} & {r['n']} & {r['rep']:.3f} {fci(r['rep_ci'])} & "
                     f"{r['free']:.3f} {fci(r['free_ci'])} & {d} {fci(r['delta_ci'])} & "
                     f"{fp(r['p_wilcoxon']).replace('<', '$<$')} & {fp(r['p_sign']).replace('<', '$<$')} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\par\smallskip\parbox{{\linewidth}}{{\footnotesize Free-run distance = mean JSD to all {n_free} free-run dyads. "
              rf"Primary inference: paired bootstrap ({B:,} resamples) 95\% CI on $\Delta$; bold = excludes 0. "
              r"Secondary: Wilcoxon signed-rank; floor: sign test. $\Delta$Distributive is an absolute error, not a JSD.}",
              rf"\caption{{Distance to the human dyads, replay vs free run --- {t(title)} (Meister et al. 2025, Eq. 1). "
              r"Lower is closer; $\Delta>0$ means replay is closer.}",
              r"\end{table*}"]                                              # ACL: caption below the table
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures (inline SVG; same palette as the other reports)
# ---------------------------------------------------------------------------

C_REPLAY, C_FREE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
C_INK, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#dedcd3"


def _nice_ticks(lo: float, hi: float, n: int = 5):
    span = hi - lo if hi > lo else 1.0
    raw = span / n
    mag = 10 ** np.floor(np.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=mag * 10)
    start, end = np.floor(lo / step) * step, np.ceil(hi / step) * step      # always cover [lo, hi]
    return [round(t, 10) for t in np.arange(start, end + step * 0.5, step)]


def _tick(t: float) -> str:
    """.030 / −.030 / 0, as in a journal axis."""
    if abs(t) < 1e-12:
        return "0"
    return f"{t:.3f}".replace("0.", ".").replace("-", "−")


def forest_table_svg(rows, sid: str = "mfig") -> str:
    """One row per configuration, drawn at its printed size: ACL two-column width (455 pt),
    text 6.5-8 pt. Columns: replay and free-run distances, Δ = free − replay with its 95%
    paired-bootstrap CI, the same as text, Wilcoxon p. Whether the CI clears 0 is shown by
    marker SHAPE (filled vs hollow) and line weight, not colour alone, so it survives grayscale.
    rows: [{arch, model, rep, free, delta, delta_ci, excl0, p_w}]."""
    import paper_figures as PF
    W = PF.TEXT_PT
    FS, FS_S, FS_H = 8, 6.5, 6.2                         # body / secondary / column-head sizes, pt
    row_h, top = 17, 24
    lab_x, rep_x, free_x, px0, px1, d_x, p_x = 92, 112, 150, 178, 320, 330, W - 2   # p column right-aligned
    lo = min(min(r["delta_ci"][0] for r in rows), 0.0)
    hi = max(max(r["delta_ci"][1] for r in rows), 0.0)
    ticks = _nice_ticks(lo, hi, n=5)
    X = lambda v: px0 + (px1 - px0) * (v - ticks[0]) / (ticks[-1] - ticks[0])
    head = f"font-size='{FS_H}' letter-spacing='0.3' fill='{C_MUTED}'"
    yb = top + row_h * len(rows)
    s = [f"<text x='{rep_x}' y='{top - 5}' text-anchor='middle' {head}>REPLAY</text>",
         f"<text x='{free_x}' y='{top - 5}' text-anchor='middle' {head}>FREE RUN</text>",
         f"<text x='{X(0) + 4:.1f}' y='{top - 5}' {head}>Δ = FREE RUN − REPLAY</text>",   # clear of the zero line
         f"<text x='{X(0):.1f}' y='{top - 14}' text-anchor='middle' font-size='{FS_H}' fill='{C_MUTED}'>no difference</text>",
         f"<text x='{d_x}' y='{top - 5}' {head}>Δ [95% CI]</text>",
         f"<text x='{p_x}' y='{top - 5}' text-anchor='end' {head}>WILCOXON p</text>"]
    for i in range(len(rows)):
        if i % 2:
            s.append(f"<rect x='0' y='{top + row_h * i}' width='{W}' height='{row_h}' fill='#f2f1ed'/>")
    for t in ticks:
        s.append(f"<line x1='{X(t):.2f}' y1='{top}' x2='{X(t):.2f}' y2='{yb}' stroke='{C_GRID}' stroke-width='0.5'/>")
        s.append(f"<text x='{X(t):.2f}' y='{yb + 8}' text-anchor='middle' font-size='{FS_S}' fill='{C_INK}'>{_tick(t)}</text>")
    s.append(f"<line x1='{X(0):.2f}' y1='{top - 10}' x2='{X(0):.2f}' y2='{yb}' stroke='{C_INK}' stroke-width='0.8'/>")
    for i, r in enumerate(rows):
        y = top + row_h * i + row_h / 2
        lo_, hi_ = r["delta_ci"]
        delta_txt = f"{r['delta']:+.3f}".replace("-", "−")
        ci_txt = f"[{lo_:.3f}, {hi_:.3f}]".replace("-", "−")
        # clears 0: dark, solid 1 pt whisker, filled dot; spans 0: light, 0.6 pt dashed whisker, hollow dot
        col, lw, dash = (C_REPLAY, 1.0, "") if r["excl0"] else ("#7fa9df", 0.7, " stroke-dasharray='1.6,1'")
        tip = (f"{r['arch']} · {r['model']}: replay {r['rep']:.3f}, free run {r['free']:.3f}; Δ {delta_txt}, "
               f"95% CI {ci_txt}; Wilcoxon p {fp(r['p_w'])}")
        dot = (f"<circle cx='{X(r['delta']):.2f}' cy='{y:.2f}' r='2.8' fill='{C_REPLAY}' stroke='#ffffff' stroke-width='0.6'/>"
               if r["excl0"] else
               f"<circle cx='{X(r['delta']):.2f}' cy='{y:.2f}' r='2.6' fill='#ffffff' stroke='{C_REPLAY}' stroke-width='1'/>")
        s.append(f"<g><title>{html.escape(tip)}</title>"
                 f"<rect x='0' y='{y - row_h / 2:.1f}' width='{W}' height='{row_h}' fill='transparent'/>"
                 f"<text x='{lab_x}' y='{y - 0.5:.2f}' text-anchor='end' font-size='{FS}' fill='{C_INK}'>{html.escape(r['arch'])}</text>"
                 f"<text x='{lab_x}' y='{y + 6.3:.2f}' text-anchor='end' font-size='{FS_S - 0.5}' fill='{C_MUTED}'>{html.escape(r['model'])}</text>"
                 f"<text x='{rep_x}' y='{y + 2.7:.2f}' text-anchor='middle' font-size='{FS}' fill='{C_INK}'>{r['rep']:.3f}</text>"
                 f"<text x='{free_x}' y='{y + 2.7:.2f}' text-anchor='middle' font-size='{FS}' fill='{C_INK}'>{r['free']:.3f}</text>"
                 f"<line x1='{X(lo_):.2f}' y1='{y:.2f}' x2='{X(hi_):.2f}' y2='{y:.2f}' stroke='{col}' stroke-width='{lw}'{dash}/>"
                 f"<line x1='{X(lo_):.2f}' y1='{y - 2.5:.2f}' x2='{X(lo_):.2f}' y2='{y + 2.5:.2f}' stroke='{col}' stroke-width='{lw}'/>"
                 f"<line x1='{X(hi_):.2f}' y1='{y - 2.5:.2f}' x2='{X(hi_):.2f}' y2='{y + 2.5:.2f}' stroke='{col}' stroke-width='{lw}'/>"
                 f"{dot}"
                 f"<text x='{d_x}' y='{y + 2.7:.2f}' font-size='{FS}' fill='{C_INK}'>"
                 f"<tspan font-weight='{700 if r['excl0'] else 400}'>{delta_txt}</tspan>"
                 f"<tspan fill='{C_MUTED}' font-size='{FS_S}'> {ci_txt}</tspan></text>"
                 f"<text x='{p_x}' y='{y + 2.7:.2f}' text-anchor='end' font-size='{FS}' fill='{C_INK}'>{fp(r['p_w']).replace('0.', '.')}</text></g>")
    s.append(f"<text y='{yb + 18}' font-size='{FS_S + 0.5}' fill='{C_INK}'>"
             f"<tspan x='{X(0) - 5:.2f}' text-anchor='end'>← free run closer</tspan>"
             f"<tspan x='{X(0) + 5:.2f}' text-anchor='start'>replay closer →</tspan></text>")
    ly = yb + 31
    s.append(f"<circle cx='{px0 + 3}' cy='{ly - 2.3}' r='2.8' fill='{C_REPLAY}'/>"
             f"<text x='{px0 + 9}' y='{ly}' font-size='{FS_S}' fill='{C_INK}'>interval clears 0 (filled, solid)</text>"
             f"<circle cx='{px0 + 112}' cy='{ly - 2.3}' r='2.6' fill='#ffffff' stroke='{C_REPLAY}' stroke-width='1'/>"
             f"<line x1='{px0 + 104}' y1='{ly - 2.3}' x2='{px0 + 109.4}' y2='{ly - 2.3}' stroke='#7fa9df' stroke-width='0.7' stroke-dasharray='1.6,1'/>"
             f"<text x='{px0 + 118}' y='{ly}' font-size='{FS_S}' fill='{C_INK}'>interval spans 0 (hollow, dashed)</text>")
    return PF.svg_root("".join(s), W, ly + 4, sid,
                       "Difference in distance to the human dyads, free run minus replay, per configuration")


def figure_block(allres, n_human: int, B: int, B_check: int, fig_no: int = 1) -> tuple[str, dict, str]:
    """The report's one figure: every configuration with both designs, sorted by Δ.
    Returns (html, {file stem: svg}, LaTeX figure snippet)."""
    import paper_figures as PF
    eq = "Eq. 1 aggregate (Pre, Post)"
    rows = [{"arch": AR.arch_label(arch), "model": AR.model_label(mdl), "rep": res[eq]["rep"],
             "free": res[eq]["free"], "delta": res[eq]["delta"], "delta_ci": res[eq]["delta_ci"],
             "excl0": res[eq]["excl0"], "p_w": res[eq]["p_wilcoxon"], "n_post": res["Post-BATNA mix"]["n"]}
            for _, _, res, arch, mdl in allres]
    if not rows:
        return "", {}, ""
    rows.sort(key=lambda r: -r["delta"])
    pos = sum(r["excl0"] and r["delta"] > 0 for r in rows)
    neg = sum(r["excl0"] and r["delta"] < 0 for r in rows)
    if pos > len(rows) / 2 and not neg:
        title = "Replay is closer to the human dyads than free run"
    elif neg > len(rows) / 2 and not pos:
        title = "Free run is closer to the human dyads than replay"
    else:
        title = "Replay vs free run: distance to the human dyads"
    svg = forest_table_svg(rows)
    caption = (f"{title}. Each row is one configuration (architecture, with the backbone model beneath). "
               f"Distance = mean per-dyad Jensen–Shannon divergence between a human dyad's behaviour mix over "
               f"Neutral / Integrative / Distributive and the simulated mix, averaged over the Pre- and Post-BATNA "
               f"phases with equal weight (Meister et al., 2025, Eq. 1); lower is closer to humans. Replay compares "
               f"each human dyad with its own replay; free run with the average over the free-run dyads. "
               f"Δ = free run − replay, so Δ > 0 means replay is closer. Intervals are 95% percentile CIs from a "
               f"paired bootstrap over the human dyad ids ({B:,} resamples; stability checked at {B_check:,}); filled "
               f"markers and solid whiskers mark intervals that clear 0, hollow markers and dashed whiskers intervals "
               f"that span it. p is a two-sided Wilcoxon signed-rank test on the per-dyad differences. {n_human} human "
               f"dyads; Post-BATNA uses the {rows[0]['n_post']} reaching both phases. Replay receives the real "
               f"partner's turns as context and free run does not, so part of every difference is that conditioning.")
    buttons = ("<div class='figlegend'><span class='muted'>One interval per row — nothing to read by overlap. "
               "Vector PDF: figures/meister_fig_forest.pdf</span><span class='grow'></span>"
               "<button onclick=\"downloadChart('mfig', 'meister_fig_forest', 'png')\">Download PNG</button>"
               "<button onclick=\"downloadChart('mfig', 'meister_fig_forest', 'svg')\">Download SVG</button></div>")
    html_ = f"""<script>{PF.DOWNLOAD_SCRIPT}</script>
<h2 class="figtitle">{html.escape(title)}</h2>
<div class="lede">One row per configuration. The dot is the difference in distance to the real negotiators, free run
minus replay, with its 95% paired-bootstrap interval. <b>A bar that clears the zero line is a difference the data
supports.</b> The two distance columns give the levels the difference is taken between.</div>
<div class="figcard">{PF.figure_html(svg, fig_no, html.escape(caption), buttons)}</div>"""
    tex = PF.tex_figure("figures/meister_fig_forest.pdf", PF.tex_text(caption), wide=True, label="fig:meister-forest")
    return html_, {"meister_fig_forest": svg}, tex


import paper_figures as _PF
CSS = K.CSS + _PF.CSS + """
table.mt td, table.mt th { padding: 4px 10px; }
table.mt tr:first-child th { border-top: 1.5px solid #000; border-bottom: 0.8px solid #000; font-weight: 600; }
table.mt tr:last-child td { border-bottom: 1.5px solid #000; }
table.mt tr.sub td { color: #52514e; }
.ci { font-size: 11px; color: #52514e; }
h2 { font-family: -apple-system, sans-serif; font-size: 16px; margin-top: 44px; }
table.sum td, table.sum th { padding: 4px 10px; }
.lede { font-family: -apple-system, sans-serif; font-size: 14px; color: #3d3c39; max-width: 900px; line-height: 1.5; }
.figcard { border: 1px solid #dedcd3; border-radius: 12px; padding: 16px 20px; background: #fff; max-width: 920px; margin-top: 14px; }
.figlegend { display: flex; align-items: center; gap: 18px; border-top: 1px solid #e6e4dc; padding-top: 12px;
  font-family: -apple-system, sans-serif; font-size: 14px; }
.figlegend i { display: inline-block; width: 14px; height: 14px; border-radius: 3px; margin-right: 6px; vertical-align: -2px; }
.figlegend .muted { color: #6b6a65; }
.figlegend .grow { flex: 1; }
.figlegend button { font: inherit; font-size: 13px; padding: 5px 12px; border: 1px solid #dedcd3; border-radius: 8px;
  background: #fcfcfb; cursor: pointer; }
.callout { margin-top: 14px; border: 1px solid #dedcd3; border-radius: 10px; padding: 12px 16px; background: #fafaf8;
  font-family: -apple-system, sans-serif; font-size: 13.5px; color: #3d3c39; line-height: 1.5; }
h2.figtitle { font-size: 22px; margin-bottom: 6px; }
.figfoot { font-family: -apple-system, sans-serif; font-size: 12.5px; color: #6b6a65; max-width: 920px; margin-top: 12px; line-height: 1.5; }
table.figtab { margin-top: 12px; }
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--B", type=int, default=B_MAIN, help="bootstrap resamples reported (default 1,000)")
    ap.add_argument("--check-B", type=int, default=B_CHECK, help="stability-check resamples (default 10,000)")
    ap.add_argument("--human", default=AR.HUMAN_CSV)
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)

    with contextlib.redirect_stdout(io.StringIO()):
        human = K.human_baseline_47(a.human)
    entries = AR.load_catalog()
    pairs = {}
    for e in entries:
        pairs.setdefault((e.architecture, e.model), {})[e.design] = e
    todo, skipped = [], []
    for (arch, model), d in sorted(pairs.items(), key=lambda kv: (AR.order_key("architecture", kv[0][0]),
                                                                   AR.order_key("model", kv[0][1]))):
        title = f"{AR.arch_label(arch)} · {AR.model_label(model)}"
        ok = all(k in d and os.path.exists(os.path.join(ROOT, d[k].path)) for k in ("replay", "freerun"))
        (todo if ok else skipped).append((title, d))

    sections, texs, summary, allres = [], [], [], []
    for idx, (title, d) in enumerate(todo, start=1):
        replay = AR.load_entry(d["replay"])
        free = AR.load_entry(d["freerun"])
        # seed keyed to the configuration, not its position: adding rows never moves existing CIs
        key = f"{d['replay'].architecture}|{d['replay'].model}".encode()
        res = analyse(human, replay, free, a.B, a.check_B, SEED + zlib.crc32(key) % 100_000)
        sections.append(section(idx, title, res, len(free), a.B, a.check_B))
        texs.append(tex_section(title, res, len(free), a.B))
        r = res["Eq. 1 aggregate (Pre, Post)"]
        summary.append((idx, title, r))
        allres.append((idx, title, res, d["replay"].architecture, d["replay"].model))
        print(f"{idx}. {title:38s} Eq.1 replay {r['rep']:.3f}  free {r['free']:.3f}  "
              f"Δ {r['delta']:+.3f} {fci(r['delta_ci'])}{'  CI excludes 0' if r['excl0'] else ''}  "
              f"(10k-resample CI shift {r['ci_shift']:.4f})")

    sum_rows = "".join(
        f"<tr><td class='rowh'><a href='#t{i}'>{html.escape(t)}</a></td><td>{r['rep']:.3f}</td><td>{r['free']:.3f}</td>"
        f"<td>{('<b>' if r['excl0'] else '')}{r['delta']:+.3f}{('</b>' if r['excl0'] else '')} "
        f"<span class='ci'>{fci(r['delta_ci'])}</span></td><td>{fp(r['p_wilcoxon'])}</td><td>{fp(r['p_sign'])}</td></tr>"
        for i, t, r in summary)
    skipped_note = (f"<div class='note'>Not compared (needs both a replay and a free-run result): "
                    f"{html.escape('; '.join(t for t, _ in skipped))}.</div>" if skipped else "")
    figures_html, svgs, fig_tex = figure_block(allres, len(human), a.B, a.check_B)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Replay vs free run (Meister Eq. 1)</title>
<style>{CSS}</style></head><body>
<h1>Replay vs free run — distance to human dyads (Meister et al. 2025, Eq. 1)</h1>
<div class="note">Per human dyad and phase, the distance between the human dyad's behaviour mix and the model's:
replay = its own replay; free run = the average over that free run's dyads (free runs mirror no human GameId).
Equal weight per phase. Δ = A(free run) − A(replay); Δ &gt; 0 means replay is closer to humans. Primary inference:
paired-bootstrap 95% CI on Δ ({a.B:,} resamples, stability-checked at {a.check_B:,}); Wilcoxon signed-rank and the
sign test are secondary. {len(human)} human dyads (BATNA change delivered). Rows: report/results_catalog.json.</div>
<div class="note warn">Read Δ with the design difference in mind: a replay dyad answers the actual human
conversation, a free-run dyad never sees it, so part of any replay advantage is that conditioning.</div>
{skipped_note}
<table class="kodis mt sum"><tr><th class="rowh">Comparison</th><th>Replay</th><th>Free run</th>
<th>Δ = free − replay [95% CI]</th><th>Wilcoxon p</th><th>sign-test p</th></tr>{sum_rows}</table>
<div class="tcap">Table 1: Summary — Eq. 1 aggregate distance to the human dyads per configuration, replay vs free run
(bold Δ = paired-bootstrap 95% CI excludes 0). Details per configuration in Tables 2–{len(summary) + 1}.</div>
{figures_html}
{''.join(sections)}
</body></html>"""
    os.makedirs(AR.OUT_DIR, exist_ok=True)
    out = os.path.join(AR.OUT_DIR, "meister_replay_vs_freerun.html")
    open(out, "w").write(page)
    open(out[:-5] + ".tex", "w").write("\n\n".join(([fig_tex] if fig_tex else []) + texs) + "\n")
    import glob
    import paper_figures as PF
    for old in glob.glob(os.path.join(AR.OUT_DIR, "meister_fig*.svg")):   # earlier layouts kept figures here
        os.remove(old)
    PF.export(svgs, os.path.join(AR.OUT_DIR, "figures"))          # vector PDF + SVG at printed size
    print(f"-> {os.path.relpath(out, ROOT)}" + (f"   (skipped: {len(skipped)})" if skipped else ""))
    if a.open:
        os.system(f'open "{out}"')


if __name__ == "__main__":
    main()
