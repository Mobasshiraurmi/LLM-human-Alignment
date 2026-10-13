"""kodis_stats.py — the KODIS-style divergence-gap engine, ported from Urmi_assmebledall's
kodis_style_table.py and decoupled from that branch's RAG/debate-agent ecosystem (it depended
on compare_single_vs_debate.py, report_single_agent.py, her own step5_replay.py/rag_align.py --
none of those exist on this branch). The statistics themselves are unchanged; only the data
loaders below are rewritten against this repo's own rag_align.py/step5_replay.py.

Gaps are computed as in the KODIS evaluation scripts (run_evaluation.py / strategic_behavior.py),
on our alignment metrics, with agents as rows; the TESTS are corrected, and the table reports
direction, scale and dispersion:

  ‡ distribution-level metrics ("Within-Human")
        HH  = metric over every pair of human dyads           (within-group, all pairs)
        HM  = metric over every (agent dyad, human dyad) pair  (between-group, cross product)
        AA  = metric over every pair of agent dyads            (within-agent dispersion)
        signed gap = mean(HM) - mean(HH)   (KODIS reports |.|; the sign matters: negative
                     means the agent sits closer to humans than humans sit to each other,
                     i.e. under-dispersion, flagged ↓ and never counted as "best")
        test: PERMUTATION OF DYADS (all pairwise distances computed once, group labels
              permuted, gap recomputed). The KODIS t-test over the pairwise values treats
              ~1,000-2,000 dependent pairs as independent (each dyad is in ~46 pairs) and
              gave 43-50% false positives at p<.05 on human-vs-human splits of these data;
              the permutation test gives 3.5-6.5%. Replay rows share GameIds with the human
              dyads, so labels are only swapped within each matched pair (restricted
              permutation). ANOSIM/PERMANOVA logic: Clarke 1993; Anderson 2001; Anderson &
              Walsh 2013.
  † dyad-level metrics
        signed gap = agent mean - human mean
        test: replay rows -> paired t-test on GameId (same scenario, BATNA and context),
              Wilcoxon signed-rank alongside; free-run rows -> Welch t-test (no one-to-one
              match, as in KODIS).
  Every cell also shows gap / human baseline (scale-free), and a separate dispersion table
  shows AA (‡) or the SD across dyads (†) against the human value (Kwon et al. 2025 §4.4).

As in strategic_behavior.py, a dyad's behavior distribution is its pooled message shares
(Neutral / Integrative / Distributive) and JSD is scipy.spatial.distance.jensenshannon(p, q) ** 2
(natural log).

Metrics (the ones in kodis_reports.py's tables):
  Table 1  BG-All‡  BG-Pre‡  BG-Post‡   behavior-mix JSD gap, overall / Pre- / Post-BATNA
           PPG‡                         6-D phase profile (pre mix + post mix), L2
           LenG‡                        length [words/msg / 50, msgs/dyad / 12], L2
           ΔDG†                         Distributive shift post - pre, per dyad
  Table 2  share gaps†                  Neutral / Integrative / Distributive share,
                                        Pre- and Post-BATNA, per dyad
A dyad enters a phase-specific metric only if it has messages in that phase.

Human baseline for every table: the 47 real dyads with a delivered BATNA change (the same
load_batna_targets() every other report in this repo uses). A non-significant gap is not
evidence of alignment; an equivalence claim needs TOST (Lakens 2017).
"""
from __future__ import annotations

import html
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ttest_ind, ttest_rel, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rag_align as ra

B = ra.BEHAVIORS
PRE, POST = ra.PRE, ra.POST


# ---------------------------------------------------------------------------
# Data loaders (this repo's own schema: GameId, Sender, Text, Phase, Behavior
# for the human reference CSV; GameId, Sender, Text, Phase, pred_behavior for
# any classify_replay.py/classify_freerun.py results.csv)
# ---------------------------------------------------------------------------

def load_human(path):
    h = pd.read_csv(path)
    h["Behavior"] = ra._canon_behavior(h["Behavior"])
    h["Phase"] = h["Phase"].astype(str).str.strip()
    return h.dropna(subset=["Behavior"])[["GameId", "Phase", "Text", "Behavior"]]


def load_model(path):
    m = pd.read_csv(path)
    m["Behavior"] = ra._canon_behavior(m["pred_behavior"])
    m["Phase"] = m["Phase"].astype(str).str.strip()
    return m.dropna(subset=["Behavior"])[["GameId", "Phase", "Text", "Behavior"]]


def human_baseline_47(human_csv):
    """The one human baseline every table uses: the 47 real dyads with an identifiable
    BATNA recipient, via step5_replay.load_batna_targets() -- the same function and the
    same 47 dyads every other report in this repo (report_replay.py, report_freerun.py,
    step6_paired.py) is built on."""
    import contextlib
    import io
    import step5_replay as sr
    with contextlib.redirect_stdout(io.StringIO()):
        targets = sr.load_batna_targets()
    h = load_human(human_csv)
    return per_dyad(h[h.GameId.isin(targets)])


# ---------------------------------------------------------------------------
# Per-dyad quantities (pooled message shares, as compute_strategy_distribution)
# ---------------------------------------------------------------------------

def dist(d, phase=None):
    sub = d if phase is None else d[d.Phase == phase]
    if sub.empty:
        return None
    return sub.Behavior.value_counts(normalize=True).reindex(B, fill_value=0.0).to_numpy()


def per_dyad(df):
    out = {}
    for g, d in df.groupby("GameId"):
        pre, post = dist(d, PRE), dist(d, POST)
        w = d.Text.astype(str).str.split().str.len()
        out[g] = {"all": dist(d), "pre": pre, "post": post,
                  "profile": None if pre is None or post is None else np.concatenate([pre, post]),
                  "length": np.array([w.mean() / 50.0, len(d) / 12.0]),
                  "dD": None if pre is None or post is None else post[2] - pre[2]}
        for ph, key in [(pre, "pre"), (post, "post")]:
            for i, b in enumerate(B):
                out[g][f"{key}_{b}"] = None if ph is None else ph[i]
    return out


def jsd2(p, q):
    return float(jensenshannon(p, q) ** 2)


def l2(p, q):
    return float(np.linalg.norm(p - q))


# ---------------------------------------------------------------------------
# The two procedures
# ---------------------------------------------------------------------------

PERM_B = 4999          # permutations per ‡ test
PERM_SEED = 20261004


def _perm_masks(n_h, n_a, pairs, B, rng):
    """B human-membership masks over the pooled [human..., agent...] dyads.

    Unpaired (free run): any n_h of the pooled dyads may be "human" -- plain relabelling.
    Paired (replay): agent dyad i answers human dyad i (same GameId), so only the labels
    WITHIN each matched pair are exchangeable under H0; each pair swaps with prob. 1/2
    (a restricted permutation, as for a blocked design in Anderson & Walsh 2013).
    """
    n = n_h + n_a
    if pairs:
        base = np.zeros(n, bool)
        base[:n_h] = True
        M = np.tile(base, (B, 1))
        hi = np.array([h for h, _ in pairs])
        ai = np.array([a for _, a in pairs])
        S = rng.random((B, len(pairs))) < 0.5
        M[:, hi] = ~S
        M[:, ai] = S
        return M.astype(float)
    order = np.argsort(rng.random((B, n)), axis=1)
    M = np.zeros((B, n))
    np.put_along_axis(M, order[:, :n_h], 1.0, axis=1)
    return M


def _gap_stats(D, M, n_h, n_a):
    """signed gap = mean(agent x human distance) - mean(human x human distance), per mask."""
    hh = np.einsum("bi,ij,bj->b", M, D, M) / (n_h * (n_h - 1))
    hm = np.einsum("bi,ij,bj->b", 1 - M, D, M) / (n_a * n_h)
    return hm - hh


def distribution_level(human, agent, key, fn, paired=False, B=PERM_B, seed=PERM_SEED):
    """‡: within-human pairs vs agent x human cross product, as strategic_behavior.py --
    but tested by permuting DYADS, not pairwise values.

    The KODIS script runs ttest_ind on the 1,081 HH and 2,209 HM pairwise values, which
    treats them as independent although each dyad appears in ~46 of them (the Mantel-test
    problem). On these data that gives a 43-50% false-positive rate at p<.05 when humans are
    compared with humans. Here all pairwise distances are computed once, the dyads' group
    labels are permuted (restricted to within-GameId swaps for replay rows), and the signed
    gap is recomputed each time -- the ANOSIM/PERMANOVA logic (Clarke 1993; Anderson 2001).
    Two-sided p. Also returns within-agent dispersion (mean AA distance).
    """
    hg = [g for g, v in human.items() if v[key] is not None]
    ag = [g for g, v in agent.items() if v[key] is not None]
    X = [human[g][key] for g in hg] + [agent[g][key] for g in ag]
    n_h, n_a, n = len(hg), len(ag), len(hg) + len(ag)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = fn(X[i], X[j])
    pos = {g: i for i, g in enumerate(hg)}
    pairs = [(pos[g], n_h + j) for j, g in enumerate(ag) if g in pos] if paired else []
    obs_mask = np.zeros((1, n))
    obs_mask[0, :n_h] = 1
    obs = float(_gap_stats(D, obs_mask, n_h, n_a)[0])
    rng = np.random.default_rng(seed)
    perm = _gap_stats(D, _perm_masks(n_h, n_a, pairs, B, rng), n_h, n_a)
    p = float((np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (B + 1))
    hh = float(D[:n_h, :n_h].sum() / (n_h * (n_h - 1)))
    aa = float(D[n_h:, n_h:].sum() / (n_a * (n_a - 1))) if n_a > 1 else float("nan")
    return {"base": hh, "between": hh + obs, "signed": obs, "gap": abs(obs),
            "rel": obs / hh if hh else float("nan"), "within": aa, "within_ratio": aa / hh,
            "p": p, "test": "paired dyad permutation" if pairs else "dyad permutation",
            "stat": obs, "n": (n_h, n_a), "n_pairs": len(pairs)}


def dyad_level(human, agent, key, paired=False):
    """†: per-dyad values. signed gap = agent mean - human mean.

    Replay rows (paired=True) share GameId with the human dyads -- same scenario, BATNA
    and context -- so they are tested PAIRED: paired t-test on matched GameIds, Wilcoxon
    signed-rank alongside. Free-run dyads have no human counterpart: Welch t-test, as in
    the KODIS scripts.
    """
    hd = {g: v[key] for g, v in human.items() if v[key] is not None}
    ad = {g: v[key] for g, v in agent.items() if v[key] is not None}
    hv, av = np.array(list(hd.values()), float), np.array(list(ad.values()), float)
    common = sorted(set(hd) & set(ad)) if paired else []
    if common:
        x = np.array([ad[g] for g in common]); y = np.array([hd[g] for g in common])
        t, p = ttest_rel(x, y)
        try:
            p_w = float(wilcoxon(x, y).pvalue)
        except ValueError:            # all differences zero
            p_w = 1.0
        test = "paired t (GameId)"
    else:
        t, p = ttest_ind(hv, av, equal_var=False)
        p_w, test = float("nan"), "Welch t"
    signed = float(av.mean() - hv.mean())
    base = float(hv.mean())
    return {"base": base, "between": float(av.mean()), "signed": signed, "gap": abs(signed),
            "rel": signed / abs(base) if abs(base) > 1e-9 else float("nan"),
            "within": float(av.std(ddof=1)), "within_ratio": float(av.std(ddof=1) / hv.std(ddof=1)),
            "base_sd": float(hv.std(ddof=1)),
            "p": float(p) if np.isfinite(p) else 1.0, "p_wilcoxon": p_w, "test": test,
            "stat": float(t) if np.isfinite(t) else 0.0, "n": (len(hv), len(av)),
            "n_pairs": len(common)}


TABLE1 = [  # (group, label, kind, key, fn, baseline text)
    ("Behavior Mix", "BG-All", "‡", "all", jsd2, "JSD"),
    ("Behavior Mix", "BG-Pre", "‡", "pre", jsd2, "JSD"),
    ("Behavior Mix", "BG-Post", "‡", "post", jsd2, "JSD"),
    ("Phase Dynamics", "PPG", "‡", "profile", l2, "L2"),
    ("Phase Dynamics", "ΔDG", "†", "dD", None, "ΔDist"),
    ("Message Length", "LenG", "‡", "length", l2, "L2"),
]
TABLE2 = [(f"{pn}-BATNA share", b, "†", f"{key}_{b}", None, f"{b} share")
          for key, pn in [("pre", "Pre"), ("post", "Post")] for b in B]


def run_table(spec, human, agents, paired=None):
    """paired: {agent name: True} for replay rows (same GameIds as the human dyads)."""
    paired = paired or {}
    res = {}
    for _, label, kind, key, fn, _ in spec:
        for name, a in agents.items():
            pr = bool(paired.get(name))
            res[(key, name)] = (distribution_level(human, a, key, fn, paired=pr) if kind == "‡"
                                else dyad_level(human, a, key, paired=pr))
    return res


# ---------------------------------------------------------------------------
# Rendering (layout of the KODIS table: grouped header, bold best, stars, baseline row)
# ---------------------------------------------------------------------------

def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def baseline_text(kind, unit):
    if kind == "‡":
        return "Avg. Within-Human", f"{unit}: "
    return "Avg. Human", f"{unit}: "


def _groups(spec):
    groups = []
    for g, *_ in spec:
        if not groups or groups[-1][0] != g:
            groups.append([g, 0])
        groups[-1][1] += 1
    return groups


def _rel(r):
    return "" if not np.isfinite(r.get("rel", np.nan)) else f"{100 * r['rel']:+.0f}%"


def _down(kind, r):
    """‡ cell whose agent-vs-human distance is BELOW the human-vs-human distance: the agent
    sits closer to the humans than humans sit to each other -- under-dispersion (e.g. a
    model that always produces the average), not alignment."""
    return kind == "‡" and r["signed"] < 0


def _best(spec_kind, key, res, agents):
    """Smallest |gap| among the cells eligible to be 'best' -- ↓ cells are excluded."""
    ok = [a for a in agents if not _down(spec_kind, res[(key, a)])]
    return min((res[(key, a)]["gap"] for a in ok), default=None)


FOOT = ("<div class='foot'>Cells: signed gap, with the gap as a share of the human baseline below it. "
        "‡ distribution-level (“Within-Human”, from pairwise comparisons over all dyads, e.g. BG, PPG, LenG): "
        "gap = mean(agent×human) − mean(human×human); positive = the agent is farther from humans than humans "
        "are from each other; <span class='down'>↓</span> negative = closer than human-to-human, i.e. less "
        "varied than humans (possible under-dispersion / mode collapse), never counted as best. "
        "† dyad-level (averaged across dyads, e.g. ΔDG, shares): gap = agent mean − human mean.<br>"
        "Tests: ‡ permutation of dyads (not of pairwise values; within-GameId swaps for replay rows), "
        "† paired t-test on GameId for replay rows, Welch t-test for free-run rows. Two-sided. "
        "* p &lt; .05, ** p &lt; .01, *** p &lt; .001. Bold = smallest |gap| among rows with results, "
        "excluding ↓ cells. Hover a cell for the means, test, p and n.</div>")


def html_table(spec, res, agents, caption, n_h, order=None, pending=(), sublabels=None, base_name=None):
    """`agents` = rows with results; `order` = every row in display order, where names in
    `pending` (planned rows whose results CSV does not exist yet) are drawn greyed with
    'pending' cells."""
    order = list(order or agents)
    sublabels = sublabels or {}
    base_name = base_name or next(iter(agents))
    groups = _groups(spec)
    top = "".join(f"<th colspan='{n}' class='grp'>{html.escape(g)}</th>" for g, n in groups)
    head = "".join(f"<th>{html.escape(l)}<sup>{k}</sup></th>" for _, l, k, *_ in spec)
    body = ""
    for name in order:
        tag = f"<br><span class='sub'>{html.escape(sublabels[name])}</span>" if name in sublabels else ""
        if name in pending:
            cells = "".join("<td class='pending'>pending</td>" for _ in spec)
            body += f"<tr class='pending'><td class='rowh'>{html.escape(name)}{tag}</td>{cells}</tr>"
            continue
        cells = ""
        for _, label, kind, key, *_ in spec:
            r = res[(key, name)]
            best = _best(kind, key, res, agents)
            v = f"{r['signed']:+.3f}"
            if best is not None and len(agents) > 1 and np.isclose(r["gap"], best) and not _down(kind, r):
                v = f"<b>{v}</b>"
            arrow = "<span class='down'>↓</span>" if _down(kind, r) else ""
            wil = f", Wilcoxon p={r['p_wilcoxon']:.2g}" if np.isfinite(r.get("p_wilcoxon", np.nan)) else ""
            cells += (f"<td title='agent {r['between']:.3f} vs human {r['base']:.3f}; {r['test']} "
                      f"p={r['p']:.2g}{wil}; n human/agent={r['n'][0]}/{r['n'][1]}"
                      f"{', pairs=' + str(r['n_pairs']) if r['n_pairs'] else ''}'>"
                      f"{v}{arrow}{stars(r['p'])}<br><span class='rel'>{_rel(r)}</span></td>")
        body += f"<tr><td class='rowh'>{html.escape(name)}{tag}</td>{cells}</tr>"
    l1 = l2_ = ""
    for _, label, kind, key, _, unit in spec:
        a, b = baseline_text(kind, unit)
        l1 += f"<td>{a}</td>"
        l2_ += f"<td>{b}{res[(key, base_name)]['base']:.3f}</td>"
    body += (f"<tr class='base1'><td class='rowh'>Human</td>{l1}</tr>"
             f"<tr class='base2'><td class='rowh'>(Baseline, {n_h} dyads)</td>{l2_}</tr>")
    return (f"<table class='kodis'><tr class='r0'><th rowspan='3' class='rowh'>Agent</th>"
            f"<th colspan='{len(spec)}'>Metrics</th></tr><tr class='r1'>{top}</tr>"
            f"<tr class='r2'>{head}</tr>{body}</table>{FOOT}<div class='cap'>{caption}</div>")


def html_dispersion(spec, res, agents, caption, order=None, pending=(), sublabels=None, base_name=None):
    """Within-group dispersion: is each agent as VARIED as the humans? (Kwon et al. 2025,
    §4.4: human dyads vary a lot, LLM dyads much less.) ‡: mean pairwise distance among the
    agent's own dyads vs among human dyads; †: SD across dyads. Ratio < 1 = less varied."""
    order = list(order or agents)
    sublabels = sublabels or {}
    base_name = base_name or next(iter(agents))
    groups = _groups(spec)
    top = "".join(f"<th colspan='{n}' class='grp'>{html.escape(g)}</th>" for g, n in groups)
    head = "".join(f"<th>{html.escape(l)}<sup>{k}</sup></th>" for _, l, k, *_ in spec)
    def hcell(kind, r):
        return r["base"] if kind == "‡" else r["base_sd"]
    body = ("<tr><td class='rowh'>Human<br><span class='sub'>within-human</span></td>"
            + "".join(f"<td>{hcell(kind, res[(key, base_name)]):.3f}</td>" for _, _, kind, key, *_ in spec)
            + "</tr>")
    for name in order:
        tag = f"<br><span class='sub'>{html.escape(sublabels[name])}</span>" if name in sublabels else ""
        if name in pending:
            cells = "".join("<td class='pending'>pending</td>" for _ in spec)
            body += f"<tr class='pending'><td class='rowh'>{html.escape(name)}{tag}</td>{cells}</tr>"
            continue
        cells = ""
        for _, _, kind, key, *_ in spec:
            r = res[(key, name)]
            ratio = r["within_ratio"]
            low = "<span class='down'>↓</span>" if np.isfinite(ratio) and ratio < 1 else ""
            cells += (f"<td>{r['within']:.3f}{low}<br><span class='rel'>×{ratio:.2f} of human</span></td>"
                      if np.isfinite(ratio) else "<td>–</td>")
        body += f"<tr><td class='rowh'>{html.escape(name)}{tag}</td>{cells}</tr>"
    return (f"<table class='kodis'><tr class='r0'><th rowspan='3' class='rowh'>Group</th>"
            f"<th colspan='{len(spec)}'>Within-group dispersion</th></tr><tr class='r1'>{top}</tr>"
            f"<tr class='r2'>{head}</tr>{body}</table>"
            "<div class='foot'>‡ mean pairwise distance among a group's own dyads (agent×agent, human×human); "
            "† standard deviation across dyads. ×ratio = agent ÷ human; <span class='down'>↓</span> below 1 = "
            "the agent's dyads vary less than human dyads do. A row can show a small gap above while being "
            "far less varied than humans here -- read the two tables together.</div>"
            f"<div class='cap'>{caption}</div>")


def tex_table(spec, res, agents, caption, n_h, order=None, pending=(), base_name=None):
    order = list(order or agents)
    base_name = base_name or next(iter(agents))
    groups = _groups(spec)
    cols = len(spec)
    tex = lambda s: s.replace("Δ", r"$\Delta$").replace("%", r"\%")
    lines = [r"\begin{table*}[t]", r"\centering", r"\begin{tabular}{l" + "c" * cols + "}", r"\toprule",
             rf"\multirow{{3}}{{*}}{{Agent}} & \multicolumn{{{cols}}}{{c}}{{Metrics}} \\",
             rf"\cmidrule(lr){{2-{cols + 1}}}"]
    lines.append(" & " + " & ".join(rf"\multicolumn{{{n}}}{{c}}{{{tex(g)}}}" for g, n in groups) + r" \\")
    c, rules = 2, []
    for _, n in groups:
        rules.append(rf"\cmidrule(lr){{{c}-{c + n - 1}}}")
        c += n
    lines.append(" ".join(rules))
    mark = {"‡": r"$^\ddagger$", "†": r"$^\dagger$"}
    lines.append(" & " + " & ".join(tex(l) + mark[k] for _, l, k, *_ in spec) + r" \\")
    lines.append(r"\midrule")
    for name in order:
        if name in pending:
            lines.append(tex(name) + " & " + " & ".join(["--"] * cols) + r" \\")
            continue
        row = []
        for _, label, kind, key, *_ in spec:
            r = res[(key, name)]
            best = _best(kind, key, res, agents)
            v = f"{r['signed']:+.3f}"
            if best is not None and len(agents) > 1 and np.isclose(r["gap"], best) and not _down(kind, r):
                v = rf"\textbf{{{v}}}"
            if _down(kind, r):
                v += r"$\downarrow$"
            if stars(r["p"]):
                v += rf"$^{{{stars(r['p'])}}}$"
            rel = _rel(r)
            row.append(v + (rf" ({tex(rel)})" if rel else ""))
        lines.append(tex(name) + " & " + " & ".join(row) + r" \\")
    lines.append(r"\midrule")
    lines.append("Human & " + " & ".join(baseline_text(k, u)[0] for _, _, k, *_, u in spec) + r" \\")
    lines.append(rf"(Baseline, {n_h} dyads) & " + " & ".join(
        tex(baseline_text(k, u)[1]) + f"{res[(key, base_name)]['base']:.3f}"
        for _, l, k, key, _, u in spec) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{tex(caption)}}}", r"\end{table*}"]
    return "\n".join(lines)


CSS = """
body { font-family: 'Times New Roman', Times, serif; margin: 32px; color: #0b0b0b; background: #fff; }
h1 { font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 18px; }
.note { font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 12px; color: #52514e; max-width: 1100px; }
.warn { color: #d03b3b; }
table.kodis { border-collapse: collapse; margin: 28px 0 6px; font-size: 15px; }
table.kodis th, table.kodis td { padding: 3px 14px; text-align: center; font-weight: normal; }
table.kodis th.rowh, table.kodis td.rowh { text-align: left; }
table.kodis tr.r0 th { border-top: 1.5px solid #000; }
table.kodis tr.r0 th:not(.rowh), table.kodis tr.r1 th { border-bottom: 0.8px solid #000; }
table.kodis tr.r2 th { border-bottom: 0.8px solid #000; }
table.kodis tr.base1 td { border-top: 0.8px solid #000; }
table.kodis tr.base2 td { border-bottom: 1.5px solid #000; }
.foot { font-size: 13px; max-width: 1100px; line-height: 1.45; }
.cap { font-size: 17px; margin: 14px 0 34px; max-width: 1100px; text-align: center; }
table.kodis .sub { font-size: 11px; color: #52514e; }
table.kodis tr.pending td { color: #9a988f; font-style: italic; }
table.kodis .rel { font-size: 11px; color: #52514e; }
table.kodis .down { color: #b25c00; }
table.det { border-collapse: collapse; font-family: -apple-system, sans-serif; font-size: 12px; }
table.det th, table.det td { border: 1px solid #dedcd3; padding: 4px 8px; text-align: right; }
table.det th { background: #2c3e50; color: #fff; }
table.det td:first-child, table.det td:nth-child(2) { text-align: left; }
"""

def detail_rows(spec, res, agents):
    rows = ""
    for group, label, kind, key, *_ in spec:
        for name in agents:
            r = res[(key, name)]
            wil = f"{r['p_wilcoxon']:.2g}" if np.isfinite(r.get("p_wilcoxon", np.nan)) else ""
            rows += (f"<tr><td>{html.escape(group)} · {html.escape(label)}{kind}</td><td>{html.escape(name)}</td>"
                     f"<td>{r['base']:.4f}</td><td>{r['between']:.4f}</td><td>{r['signed']:+.4f}</td>"
                     f"<td>{_rel(r)}</td><td>{r['within']:.4f} (×{r['within_ratio']:.2f})</td>"
                     f"<td>{html.escape(r['test'])}</td><td>{r['p']:.2g}</td><td>{wil}</td>"
                     f"<td>{r['n'][0]} / {r['n'][1]}{' (' + str(r['n_pairs']) + ' pairs)' if r['n_pairs'] else ''}</td></tr>")
    return rows


DETAIL_HEAD = ("<tr><th>metric</th><th>agent</th><th>human (within / mean)</th><th>agent vs human</th>"
               "<th>signed gap</th><th>gap ÷ baseline</th><th>agent within (× human)</th><th>test</th>"
               "<th>p</th><th>Wilcoxon p</th><th>n human / agent</th></tr>")


def method_note(n_h, human_csv):
    return f"""<div class="note">Procedure after the KODIS evaluation scripts (strategic_behavior.py /
run_evaluation.py), with two corrections to the tests. ‡ distribution-level metrics: within-human vs
agent-vs-human pairwise distances; tested by permuting <em>dyads</em> ({PERM_B:,} permutations; within-GameId
swaps for replay rows), not by a t-test over the pairwise values, which treats ~1,000-2,000 dependent pairs as
independent and gave 43-50% false positives on these data. † dyad-level metrics: paired t-test on GameId for replay
rows (each replay dyad answers the same real dyad), Welch t-test for free-run rows. Gaps are signed and shown
relative to the human baseline. Human baseline for every row and every table: {n_h} real dyads with an
identifiable BATNA recipient ({human_csv}). All rows are coded by the same GPT-5.4 classifier.</div>
<div class="note warn">A non-significant gap is not evidence of alignment: it means no gap was detectable at this
sample size. An equivalence claim needs an equivalence test (TOST; Lakens 2017). A ↓ cell (agent closer to the
humans than humans are to each other) indicates under-dispersion, not better alignment -- check the dispersion
table.</div>"""
