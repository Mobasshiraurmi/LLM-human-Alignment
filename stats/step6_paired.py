"""
step6_paired.py — analyse the replay results
============================================

Run after classifying replay/<agent>/replay_unlabeled.csv with classify_replay.py.

    python stats/step6_paired.py

WHAT IS DIFFERENT FROM step3/step4
    In the replay design every model turn has a HUMAN COUNTERPART at the same
    context. That unlocks paired statistics the free-running design could not
    support -- above all turn-level agreement.

WHAT IT REPORTS
    1. Behaviour mix, pre vs post, per agent               -> DeltaDistributive
    2. Turn-level agreement (Gwet's AC1)                    -> new, paired
    3. Role split: recipient vs counterpart                -> the hard test
    4. Style-vector distance (L2 / cosine, bootstrapped)    -> Li et al. units
    5. Transition-conditional behaviour                     -> path dependence
    5e. Paired contrast: multi-agent vs single-agent         -> the architecture claim
    6. Length distribution

CLUSTERING
    ~560 turns come from 47 conversations, so turns are NOT independent.
    Every bootstrap here resamples DYADS, not turns. Resampling turns would
    shrink the intervals by roughly sqrt(20) and make everything look
    significant.
"""

import ast
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # rag_align (same folder)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "classify"))  # classifier_bridge
import rag_align as ra
import classifier_bridge as cb

RUNS_DIR = "data/replay"
AGENTS = ["single", "multi"]
B = 1000
RNG = np.random.default_rng(20260811)


# ---------------------------------------------------------------------------
# Load: join classifier output back onto the pairs file
# ---------------------------------------------------------------------------

def load_pairs(agent: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (messages_df, turns_df).

    messages_df: every human message AND every model message, independently
    labeled, one row each -- human_behavior/model_behavior (and human_text/
    model_text) populated only on the rows they apply to, NaN otherwise. Every
    distributional function below (mix/delta_dist/shift_vector/style_dist/path
    dependence) reads a column via pandas' value_counts()/dropna(), which skips
    NaN automatically -- so no cross-side join is needed, and no non-final
    burst message is silently discarded from the mix (that was the bug in
    folding everything to one label per turn before computing these).

    turns_df: one row per TURN with a genuinely PAIRED human_behavior/
    model_behavior (closing message of the burst, via
    cb.fold_messages_to_turns) -- for the one thing that actually needs a
    single label per side per decision point: turn-level agreement (accuracy,
    Gwet's AC1, per-class recall).
    """
    rdir = f"{RUNS_DIR}/{agent}"
    pairs = pd.read_csv(f"{rdir}/replay_pairs.csv")
    unlab = pd.read_csv(f"{rdir}/replay_unlabeled.csv")
    res = f"{rdir}/results.csv"
    if not os.path.exists(res):
        raise FileNotFoundError(f"{res} missing — run classify/classify_replay.py --agent {agent} --parse")

    lab = cb.classifier_to_manual(res, unlab, sort_cols=("GameId", "turn_idx", "message_index"))
    lab["model_behavior"] = ra._canon_behavior(lab["Behavior"])
    turn_meta = pairs[["game_id", "turn_idx", "phase", "round", "role", "is_recipient", "prev_own"]]
    model_long = (lab.rename(columns={"GameId": "game_id"})
                  .merge(turn_meta, on=["game_id", "turn_idx"], how="left"))
    model_long["model_text"] = model_long["Text"]
    model_long["human_behavior"] = np.nan
    model_long["human_text"] = np.nan

    human_rows = []
    for _, row in pairs.iterrows():
        msgs = ast.literal_eval(row["human_messages"])
        behs = ast.literal_eval(row["human_behaviors"])
        for mi, (m, b) in enumerate(zip(msgs, behs), start=1):
            human_rows.append({
                "game_id": row["game_id"], "turn_idx": row["turn_idx"], "message_index": mi,
                "phase": row["phase"], "round": row["round"], "role": row["role"],
                "is_recipient": row["is_recipient"], "prev_own": row["prev_own"],
                "human_behavior": b, "human_text": m,
                "model_behavior": np.nan, "model_text": np.nan,
            })
    human_long = pd.DataFrame(human_rows)
    human_long["human_behavior"] = ra._canon_behavior(human_long["human_behavior"])

    cols = ["game_id", "turn_idx", "message_index", "phase", "round", "role",
            "is_recipient", "prev_own", "human_behavior", "human_text",
            "model_behavior", "model_text"]
    messages_df = pd.concat([human_long[cols], model_long[cols]], ignore_index=True)

    human_turns = cb.fold_messages_to_turns(
        human_long.rename(columns={"human_behavior": "Behavior"}), turn_keys=("game_id", "turn_idx"))
    model_turns = cb.fold_messages_to_turns(model_long, turn_keys=("game_id", "turn_idx"))
    turns_df = (human_turns[["game_id", "turn_idx", "phase", "round", "role",
                             "is_recipient", "prev_own", "Behavior", "human_text"]]
                .merge(model_turns[["game_id", "turn_idx", "Behavior", "model_text"]],
                       on=["game_id", "turn_idx"], suffixes=("_human", "_model")))
    turns_df = turns_df.rename(columns={"Behavior_human": "human_behavior",
                                        "Behavior_model": "model_behavior"})
    turns_df = turns_df.dropna(subset=["human_behavior", "model_behavior"])
    return messages_df, turns_df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mix(labels: pd.Series) -> np.ndarray:
    v = labels.value_counts().reindex(ra.BEHAVIORS).fillna(0).values.astype(float)
    return v / v.sum() if v.sum() else v


def gwet_ac1(a: pd.Series, b: pd.Series) -> float:
    """Chance-corrected agreement, robust to class imbalance.

    Cohen's kappa deflates badly when one class is rare -- the "kappa
    paradox". Distributive is ~14% of your turns, exactly that regime.
    Li et al. (2025) use AC1 for the same reason; their expert-vs-expert
    benchmark was AC1 = 0.678.

    q is the number of categories in the CODING SCHEME (3: Neutral/
    Integrative/Distributive), not however many happen to appear in `a`/`b` --
    Gwet's own definition fixes q at the scheme size. Deriving it from
    observed categories understates q whenever a small/bootstrap sample
    happens to miss one (plausible for Distributive at ~14% of turns), which
    inflates Pe and deflates AC1 for exactly those draws -- a bias that shows
    up in boot_dyads' CI even when the full-sample point estimate is fine.
    """
    a, b = a.astype(str), b.astype(str)
    po = float((a == b).mean())
    cats = ra.BEHAVIORS
    q = len(cats)
    if q < 2:
        return np.nan
    pi = [((a == c).mean() + (b == c).mean()) / 2 for c in cats]
    pe = sum(p * (1 - p) for p in pi) / (q - 1)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def boot_dyads(df: pd.DataFrame, stat, B: int = B, rng=RNG):
    """Bootstrap CI resampling whole dyads."""
    games = df.game_id.unique()
    out = []
    for _ in range(B):
        g = rng.choice(games, len(games), replace=True)
        sub = pd.concat([df[df.game_id == x] for x in g], ignore_index=True)
        try:
            v = stat(sub)
            if v is not None and np.isfinite(v):
                out.append(v)
        except Exception:
            pass
    if not out:
        return np.nan, np.nan
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def delta_dist(df: pd.DataFrame, col: str) -> float:
    pre = df[df.phase == ra.PRE]
    post = df[df.phase == ra.POST]
    if not len(pre) or not len(post):
        return np.nan
    return mix(post[col])[2] - mix(pre[col])[2]


def style_dist(df: pd.DataFrame, kind="L2") -> float:
    return ra.dist(mix(df.human_behavior), mix(df.model_behavior), kind)


def shift_vector(df: pd.DataFrame, col: str) -> np.ndarray:
    """3-D shift: post mix minus pre mix. The FULL adaptation signature,
    not just the Distributive component."""
    pre, post = df[df.phase == ra.PRE], df[df.phase == ra.POST]
    if not len(pre) or not len(post):
        return np.full(3, np.nan)
    return mix(post[col]) - mix(pre[col])


def splithalf_null_shift(df: pd.DataFrame, B: int = B, rng=RNG) -> np.ndarray:
    """How far apart are the SHIFT VECTORS of two halves of the human data?

    This is the yardstick from step1/step4, rebuilt for the replay design.
    Bootstrap CIs on Delta alone say "the model's number overlaps ours";
    this says "the model's whole adaptation signature is no further from us
    than we are from ourselves" -- a stronger and different claim.

    Splits WHOLE DYADS, never turns.
    """
    games = df.game_id.unique()
    n = len(games) // 2
    out = []
    for _ in range(B):
        sh = rng.permutation(games)
        A = df[df.game_id.isin(sh[:n])]
        Bh = df[df.game_id.isin(sh[n:2 * n])]
        va, vb = shift_vector(A, "human_behavior"), shift_vector(Bh, "human_behavior")
        if np.all(np.isfinite(va)) and np.all(np.isfinite(vb)):
            out.append(float(np.linalg.norm(va - vb)))
    return np.array(out)


def null_test_shift(df: pd.DataFrame, B: int = B, rng=RNG) -> dict:
    """Is the model's shift vector inside human-to-human variation?

    Both sides are matched in size: n human dyads vs n model dyads, resampled
    B times. Without size-matching the model side would carry less sampling
    noise and everything would look falsely aligned.
    """
    null = splithalf_null_shift(df, B=B, rng=rng)
    games = df.game_id.unique()
    n = len(games) // 2
    obs = []
    for _ in range(B):
        sh = rng.permutation(games)
        A = df[df.game_id.isin(sh[:n])]
        M = df[df.game_id.isin(sh[n:2 * n])]
        va, vm = shift_vector(A, "human_behavior"), shift_vector(M, "model_behavior")
        if np.all(np.isfinite(va)) and np.all(np.isfinite(vm)):
            obs.append(float(np.linalg.norm(va - vm)))
    obs = np.array(obs)
    # At small n, EVERY resample can land on a dyad subset with no Post-BATNA turns
    # on one side -- shift_vector then returns all-NaN and every draw above gets
    # skipped, leaving `null` and/or `obs` empty. np.percentile on an empty array
    # raises IndexError rather than returning nan, so guard explicitly instead of
    # letting that crash the whole report.
    if len(null) == 0 or len(obs) == 0:
        return {"d_HH_median": np.nan, "d_HH_p95": np.nan, "d_HM": np.nan,
                "ratio": np.nan, "p_perm": np.nan, "aligned": None,
                "note": f"undefined: {len(null)}/{B} null draws, {len(obs)}/{B} "
                        "obs draws had a phase with no turns on both sampled halves."}
    med = float(np.median(null))
    d_hm = float(obs.mean())
    return {"d_HH_median": med, "d_HH_p95": float(np.percentile(null, 95)),
            "d_HM": d_hm, "ratio": d_hm / med if med else np.nan,
            "p_perm": float((null >= d_hm).mean()),
            "aligned": float((null >= d_hm).mean()) > 0.05}


def shuffled_phase_bound(df: pd.DataFrame, B: int = B, rng=RNG) -> dict:
    """LOWER BOUND: what does 'no adaptation signal at all' look like?

    Socrates (Kolluri et al. 2025, Sec 5.2) anchor their scale at both ends --
    an "Empirical Best" floor from bootstrapping, and a "Uniform Guess" ceiling
    from comparing against a uniform distribution. Without the ceiling, a
    distance near the floor has no scale: the reader cannot tell whether it is
    close because the model is good or because the metric cannot discriminate.

    Our equivalent of Uniform Guess: randomly reassign the pre/post labels
    WITHIN each dyad, preserving each dyad's overall behaviour mix but
    destroying any real pre->post structure. The resulting distance is what a
    model with zero adaptation signal would produce.
    """
    games = df.game_id.unique()
    n = len(games) // 2
    out = []
    for _ in range(B):
        sh = df.copy()
        # shuffle phase labels inside each dyad
        sh["phase"] = sh.groupby("game_id")["phase"].transform(
            lambda x: rng.permutation(x.values))
        g = rng.permutation(games)
        A = df[df.game_id.isin(g[:n])]
        Msh = sh[sh.game_id.isin(g[n:2 * n])]
        va = shift_vector(A, "human_behavior")
        vm = shift_vector(Msh, "model_behavior")
        if np.all(np.isfinite(va)) and np.all(np.isfinite(vm)):
            out.append(float(np.linalg.norm(va - vm)))
    out = np.array(out)
    if len(out) == 0:
        # same empty-resample risk as null_test_shift -- see its comment.
        return {"mean": np.nan, "median": np.nan, "p05": np.nan, "p95": np.nan,
                "note": f"undefined: 0/{B} draws had a phase with turns on both sides."}
    return {"mean": float(out.mean()), "median": float(np.median(out)),
            "p05": float(np.percentile(out, 5)),
            "p95": float(np.percentile(out, 95))}


def ratio_ci(df: pd.DataFrame, B_outer: int = 400, B_inner: int = 100,
             rng=RNG) -> tuple:
    """Bootstrap CI on the ratio d_HM / d_HH_median, resampling DYADS.

    WHY THIS MATTERS
    ----------------
    A ratio of 1.09 with no interval is a point estimate -- it cannot fail.
    With an interval you get a claim that could have come out otherwise:

      if multi-agent's CI INCLUDES 1.0 and single-agent's EXCLUDES it, you can
      say that architecture's adaptation signature is as close to human dyads
      as human dyads are to each other, while the other is significantly further.

    Nested bootstrap, so it is slow: B_outer resamples of dyads, each running
    B_inner iterations of the null test. 400 x 100 is usually enough to place
    the interval; raise B_outer for the number you report.
    """
    games = df.game_id.unique()
    out = []
    for _ in range(B_outer):
        g = rng.choice(games, len(games), replace=True)
        sub = pd.concat([df[df.game_id == x] for x in g], ignore_index=True)
        try:
            r = null_test_shift(sub, B=B_inner, rng=rng)["ratio"]
            if np.isfinite(r):
                out.append(r)
        except Exception:
            pass
    if len(out) < 20:
        return (np.nan, np.nan)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def paired_delta_by_dyad(df: pd.DataFrame, col: str) -> pd.Series:
    """Per-DYAD Delta, so the shift can be tested with a t-test.

    Replay has no per-actor trajectory (each response is independent), so the
    dyad is the natural clustering unit -- unlike step4, where each actor had
    a continuous pre and post stretch.
    """
    out = {}
    for gid, g in df.groupby("game_id"):
        pre, post = g[g.phase == ra.PRE], g[g.phase == ra.POST]
        if len(pre) >= 2 and len(post) >= 2:
            out[gid] = mix(post[col])[2] - mix(pre[col])[2]
    return pd.Series(out)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def main():
    data, turns = {}, {}
    for c in AGENTS:
        try:
            data[c], turns[c] = load_pairs(c)
            print(f"{c}: {len(turns[c])} turns / {len(data[c])} messages, "
                  f"{data[c].game_id.nunique()} dyads")
        except Exception as e:
            print(f"{c}: skipped — {e}")
    if not data:
        sys.exit("Nothing to analyse.")
    any_df = next(iter(data.values()))

    # ---- 1. Delta Distributive -------------------------------------------
    print("\n" + "=" * 78)
    print("1. DELTA DISTRIBUTIVE  (post - pre)")
    print("=" * 78)
    hd = delta_dist(any_df, "human_behavior")
    lo, hi = boot_dyads(any_df, lambda d: delta_dist(d, "human_behavior"))
    print(f"  human  {hd:+.4f}  [{lo:+.4f}, {hi:+.4f}]   (paper Table 1: +0.101)")
    for c, df in data.items():
        v = delta_dist(df, "model_behavior")
        lo, hi = boot_dyads(df, lambda d: delta_dist(d, "model_behavior"))
        print(f"  {c:6s} {v:+.4f}  [{lo:+.4f}, {hi:+.4f}]   ratio {v/hd:5.2f}")
    print("\n  Every context is a real human situation, so a shift here cannot")
    print("  come from the model drifting through its own conversation.")

    # ---- 2. Turn-level agreement -----------------------------------------
    print("\n" + "=" * 78)
    print("2. TURN-LEVEL AGREEMENT  (same context, same choice?)")
    print("=" * 78)
    print(f"  {'':6s} {'acc':>7s} {'AC1':>7s} {'AC1 95% CI':>22s}")
    for c, df in turns.items():
        acc = float((df.human_behavior == df.model_behavior).mean())
        ac1 = gwet_ac1(df.human_behavior, df.model_behavior)
        lo, hi = boot_dyads(df, lambda d: gwet_ac1(d.human_behavior, d.model_behavior))
        print(f"  {c:6s} {acc:7.3f} {ac1:7.3f}   [{lo:+.3f}, {hi:+.3f}]")
    print("\n  Gwet's AC1, not Cohen's kappa: Distributive is ~14% of turns and")
    print("  kappa deflates under that skew. Li et al. expert benchmark = 0.678.")
    print("  (Paired on the TURN's closing message -- see load_pairs/turns_df.)")

    print("\n  Per-class recall (did it catch the rare class?):")
    for c, df in turns.items():
        rec = {b: float((df[df.human_behavior == b].model_behavior == b).mean())
               for b in ra.BEHAVIORS if (df.human_behavior == b).any()}
        print(f"  {c:6s} " + "  ".join(f"{k}={v:.2f}" for k, v in rec.items()))

    # ---- 3. Role split ----------------------------------------------------
    print("\n" + "=" * 78)
    print("3. ROLE SPLIT  (recipient was told; counterpart was not)")
    print("=" * 78)
    print("  A shift by the COUNTERPART can only come from reading the partner's")
    print("  changed behaviour — they never saw the update. That is the hard test.\n")
    print(f"  {'':6s} {'recipient dD':>14s} {'counterpart dD':>16s}")
    for name, col in [("human", "human_behavior")] + [(c, "model_behavior") for c in data]:
        df = any_df if name == "human" else data[name]
        r = delta_dist(df[df.is_recipient], col)
        n = delta_dist(df[~df.is_recipient], col)
        print(f"  {name:6s} {r:+14.4f} {n:+16.4f}")
    print("\n  Model shifts as recipient but not counterpart -> it reasons about")
    print("  its own payoffs but does not read its partner.")

    # ---- 4. Style distance ------------------------------------------------
    print("\n" + "=" * 78)
    print("4. STYLE-VECTOR DISTANCE  (Li et al. 2025 units)")
    print("=" * 78)
    print(f"  {'':6s} {'L2':>8s} {'95% CI':>20s} {'cosine':>9s}")
    for c, df in data.items():
        l2 = style_dist(df, "L2")
        lo, hi = boot_dyads(df, lambda d: style_dist(d, "L2"))
        cos = style_dist(df, "cosine")
        print(f"  {c:6s} {l2:8.4f}   [{lo:.4f}, {hi:.4f}] {cos:9.4f}")
    print("\n  Their best model after fine-tuning reached L2 = 0.2078 on an 8-D")
    print("  vector; yours is 3-D, so compare direction and ordering, not the")
    print("  raw value.")

    # ---- 5. Path dependence ----------------------------------------------
    print("\n" + "=" * 78)
    print("5. TRANSITION-CONDITIONAL BEHAVIOUR  (path dependence)")
    print("=" * 78)
    print("  Given the speaker's OWN previous turn, what came next?")
    print("  Your paper: beta = -0.31, so where you start shapes how much you shift.\n")
    for prev in ra.BEHAVIORS:
        sub = any_df[any_df.prev_own == prev]
        if len(sub) < 20:
            continue
        print(f"  prev={prev:13s} n={len(sub):4d}")
        print(f"    {'human':8s} {np.round(mix(sub.human_behavior), 3)}")
        for c, df in data.items():
            s = df[df.prev_own == prev]
            if len(s):
                print(f"    {c:8s} {np.round(mix(s.model_behavior), 3)}")

    # ---- 5b. Split-half null test ----------------------------------------
    print("\n" + "=" * 78)
    print("5b. IS THE MODEL'S SHIFT INSIDE HUMAN-TO-HUMAN VARIATION?")
    print("=" * 78)
    print("  Compares the FULL 3-D shift vector, not just Delta-Distributive.")
    print("  A model can match the Distributive shift while getting Neutral")
    print("  backwards -- only this test catches that.\n")
    print(f"  {'':6s} {'d_HH_med':>9s} {'d_HH_p95':>9s} {'d_HM':>8s} "
          f"{'ratio':>7s} {'ratio 95% CI':>18s} {'p':>7s}")
    for c, df in data.items():
        r = null_test_shift(df)
        lo, hi = ratio_ci(df)
        flag = ""
        if np.isfinite(lo):
            flag = "  <- CI includes 1.0" if lo <= 1.0 <= hi else "  <- CI excludes 1.0"
        print(f"  {c:6s} {r['d_HH_median']:9.4f} {r['d_HH_p95']:9.4f} "
              f"{r['d_HM']:8.4f} {r['ratio']:7.2f}   "
              f"[{lo:5.2f}, {hi:5.2f}] {r['p_perm']:7.3f}{flag}")

    print("\n  The CI is what makes this testable. A ratio of 1.09 alone is a")
    print("  point estimate that cannot fail. If one condition's CI includes 1.0")
    print("  while another's excludes it, you have a claim that could have come")
    print("  out otherwise.")
    print("\n  Do NOT rely on the p column: the baseline with no measurable")
    print("  shift also passes it, so it does not discriminate at this N.")
    print("\n  ratio 1.0 = model is exactly as far from humans as humans are")
    print("  from each other. Large p = not distinguishable at this N.")

    # ---- lower bound, so the scale has two anchors ------------------------
    lb = shuffled_phase_bound(next(iter(data.values())), B=min(B, 400))
    floor = null_test_shift(next(iter(data.values())), B=min(B, 400))["d_HH_median"]
    print("\n  SCALE ANCHORS (following Kolluri et al. 2025, Sec 5.2):")
    print(f"    within-human floor      {floor:.4f}   (their 'Empirical Best')")
    print(f"    shuffled-phase ceiling  {lb['mean']:.4f}   "
          f"(their 'Uniform Guess'; 90% range "
          f"[{lb['p05']:.4f}, {lb['p95']:.4f}])")
    print("\n    Position on that scale, as relative change from the")
    print("    no-conditioning baseline:")
    base = None
    for c, df in data.items():
        d = null_test_shift(df, B=min(B, 400))["d_HM"]
        if base is None:
            base = d
        pct = (base - d) / abs(base) * 100 if base else np.nan
        span = (lb["mean"] - floor)
        pos = (d - floor) / span * 100 if span > 0 else np.nan
        print(f"      {c:6s} d={d:.4f}  {pct:+6.1f}% vs {list(data)[0]}  "
              f"{pos:5.1f}% of the way from floor to ceiling")
    print("\n    Report RELATIVE CHANGE, not pass/fail: like the Wasserstein")
    print("    distance in Kolluri et al., this metric has a narrow range.")

    # ---- 5c. Per-dyad paired test ----------------------------------------
    print("\n" + "=" * 78)
    print("5c. PER-DYAD DELTA DISTRIBUTIVE  (paired, testable)")
    print("=" * 78)
    hd_s = paired_delta_by_dyad(any_df, "human_behavior")
    for name, ser in [("human", hd_s)] + [
            (c, paired_delta_by_dyad(df, "model_behavior")) for c, df in data.items()]:
        ser = ser.dropna()
        if len(ser) < 2:
            print(f"  {name:6s} n={len(ser)} (too few)")
            continue
        se = ser.std(ddof=1) / np.sqrt(len(ser))
        t = ser.mean() / se if se > 0 else np.nan
        # t-critical for df=n-1, not the z=1.96 normal approximation -- at the dyad
        # counts here (often well under 30), the t-distribution's fatter tails make
        # 1.96 an undercoverage CI (too narrow, overconfident).
        tcrit = stats.t.ppf(0.975, df=len(ser) - 1)
        print(f"  {name:6s} n={len(ser):3d}  mean {ser.mean():+.4f}  "
              f"SD {ser.std():.4f}  t={t:6.2f}  "
              f"95% CI [{ser.mean()-tcrit*se:+.4f}, {ser.mean()+tcrit*se:+.4f}]")

    # ---- 5d. Pre-drifted check -------------------------------------------
    print("\n" + "=" * 78)
    print("5d. IS THE MODEL 'PRE-DRIFTED'?")
    print("=" * 78)
    h_pre = mix(any_df[any_df.phase == ra.PRE].human_behavior)
    h_post = mix(any_df[any_df.phase == ra.POST].human_behavior)
    print(f"  {'':6s} {'|D_pre - human_pre|':>21s} {'|D_pre - human_post|':>22s}")
    for c, df in data.items():
        mp = mix(df[df.phase == ra.PRE].model_behavior)
        a, b = abs(mp[2] - h_pre[2]), abs(mp[2] - h_post[2])
        print(f"  {c:6s} {a:21.4f} {b:22.4f}   -> closer to "
              f"{'POST' if b < a else 'pre'}")
    print("\n  'closer to POST' means the model's PRE-perturbation behaviour")
    print("  already resembles a post-perturbation human: it starts drifted")
    print("  rather than drifting.")

    # ---- 5e. Paired contrast: multi-agent vs single-agent -----------------
    print("\n" + "=" * 78)
    print("5e. PAIRED CONTRAST  (multi-agent vs single-agent, same dyads/contexts)")
    print("=" * 78)
    print("  Both agents replay the SAME dyads and the SAME contexts (real human")
    print("  history, never their own prior output), so the difference between them")
    print("  can be bootstrapped directly -- far better powered than comparing two")
    print("  independently estimated distances. This is the core architecture claim.\n")

    games = any_df.game_id.unique()
    pairs_to_test = [("multi", "single")]
    for a, b in pairs_to_test:
        if a not in data or b not in data:
            continue
        diffs = []
        for _ in range(B):
            g = RNG.choice(games, len(games), replace=True)
            da = delta_dist(data[a][data[a].game_id.isin(g)], "model_behavior")
            db = delta_dist(data[b][data[b].game_id.isin(g)], "model_behavior")
            if np.isfinite(da) and np.isfinite(db):
                diffs.append(da - db)
        diffs = np.array(diffs)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        pv = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        star = "SIGNIFICANT" if lo > 0 or hi < 0 else "n.s."
        print(f"  {a} - {b}: {diffs.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"p={pv:.3f}  {star}")

    print("\n  This supports a DIFFERENCE claim ('multi-agent adapts more/less than")
    print("  single-agent'), NOT an equivalence claim ('either is indistinguishable")
    print("  from humans') -- see 5f for that.")

    # ---- 5f. Equivalence test --------------------------------------------
    print("\n" + "=" * 78)
    print("5f. EQUIVALENCE (TOST)  vs the human shift")
    print("=" * 78)
    print("  A non-significant difference is NOT evidence of equivalence -- it can")
    print("  just mean low power. TOST asks the right question: is the gap")
    print("  provably SMALLER than a margin you set in advance?\n")

    MARGIN = 0.05     # pre-specify this; half the human effect (0.103) is a
                      # defensible choice, but state it before looking.
    print(f"  equivalence margin = +/-{MARGIN:.3f} "
          f"(~half the human effect of {hd:+.3f})\n")
    for c, df in data.items():
        gaps = []
        for _ in range(B):
            g = RNG.choice(games, len(games), replace=True)
            sub = df[df.game_id.isin(g)]
            hh = delta_dist(sub, "human_behavior")
            mm = delta_dist(sub, "model_behavior")
            if np.isfinite(hh) and np.isfinite(mm):
                gaps.append(mm - hh)
        gaps = np.array(gaps)
        lo, hi = np.percentile(gaps, [5, 95])       # 90% CI is the TOST rule
        eq = (lo > -MARGIN) and (hi < MARGIN)
        print(f"  {c:6s} gap {gaps.mean():+.4f}  90% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"-> {'EQUIVALENT' if eq else 'not shown equivalent'}")
    print("\n  'not shown equivalent' means exactly that -- absence of evidence,")
    print("  not evidence of absence. At 47 dyads this test is demanding.")

    # ---- 6. Length --------------------------------------------------------
    print("\n" + "=" * 78)
    print("6. LENGTH")
    print("=" * 78)
    hw = any_df.human_text.dropna().astype(str).str.split().str.len()
    print(f"  human  mean {hw.mean():5.1f}  sd {hw.std():5.1f}  "
          f"<4w {(hw < 4).mean():.0%}")
    for c, df in data.items():
        mw = df.model_text.dropna().astype(str).str.split().str.len()
        print(f"  {c:6s} mean {mw.mean():5.1f}  sd {mw.std():5.1f}  "
              f"<4w {(mw < 4).mean():.0%}")
    print("\n  A model that cannot write terse acknowledgments will under-produce")
    print("  Neutral for reasons of length, not strategy. Report both together.")

    # ---- save -------------------------------------------------------------
    rows = []
    for c, df in data.items():
        t = turns[c]
        mw = df.model_text.dropna().astype(str).str.split().str.len()
        rows.append({
            "condition": c, "n_turns": len(t), "n_dyads": df.game_id.nunique(),
            "delta_D": delta_dist(df, "model_behavior"),
            "delta_D_recipient": delta_dist(df[df.is_recipient], "model_behavior"),
            "delta_D_counterpart": delta_dist(df[~df.is_recipient], "model_behavior"),
            "accuracy": float((t.human_behavior == t.model_behavior).mean()),
            "gwet_ac1": gwet_ac1(t.human_behavior, t.model_behavior),
            "style_L2": style_dist(df, "L2"),
            "style_cosine": style_dist(df, "cosine"),
            "mean_words": mw.mean(),
            "sd_words": mw.std(),
            "pct_under_4w": float((mw < 4).mean()),
            **{f"null_{k}": v for k, v in null_test_shift(df, B=400).items()},
        })
    out = pd.DataFrame(rows)
    out.to_csv("FINAL_replay.csv", index=False)
    print("\n  saved -> FINAL_replay.csv")
    print("\n  The claim lives in 5e: multi-agent vs single-agent, same dyads/contexts.")


if __name__ == "__main__":
    main()