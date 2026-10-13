"""
rag_align.py — alignment statistics: is a generated corpus's behavior close to human?
=======================================================================================

Used by every pipeline in this repo that needs to compare a generated corpus's
Neutral/Integrative/Distributive behavior mix against real human data:
report_replay.py + step6_paired.py (the replay pipeline, step5_replay.py -->
classify_replay.py, no RAG at all) and step3_classify.py + step4_adaptation.py
(the free-running live-simulation pipeline, step2_generate.py).

Also holds the RAG-retrieval machinery for the single_rag architecture
(generation/single_agent.py): build_bank, two_fold_split, retrieve,
format_exemplars, plus the validity checks copying_rate/retrieval_diagnostic
every retrieval method needs reported alongside it (ported from
Urmi_assmebledall's rag_align.py, which built the single-agent+RAG results
the paper reports). Schema-generic -- these functions only need
GameId/Round/Phase/Behavior/Text (+ an actor/role column), so they work
against the same human data every other architecture in this repo already
loads.

TARGET METRICS (static distributional only) -- 3 categories, no subtypes
    phase_profile : 6-D  [Neutral, Integrative, Distributive] x [Pre, Post] -> L2
    behavior_mix  : 3-D  overall mix                                        -> JSD
    length_vector : 2-D  [mean words, turns per dyad]                       -> L2

THE TEST
    Is dist(n humans, n model dyads) smaller than dist(n humans, n humans)?
    Both sides the same n. Resample 1000x. Report the ratio; 1.0 = model is
    exactly as far from humans as humans are from each other.

Schema (from Raiyan_analysis_2_adjudicated.ipynb):
    GameId, ProlificId, Round, Phase, Text, Behavior, BehaveSubtype
    Phase in {"Pre-BATNA", "Post-BATNA"}
    Behavior in {"Neutral", "Integrative", "Distributive"}
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd

PRE, POST = "Pre-BATNA", "Post-BATNA"
BEHAVIORS = ["Neutral", "Integrative", "Distributive"]

RNG = np.random.default_rng(20260806)


def _clean(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def _canon_behavior(s: pd.Series) -> pd.Series:
    """Canonicalize behavior labels to Title case.

    Annotation files in the wild use "neutral", "NEUTRAL", "Neutral". Without
    this, reindex(columns=BEHAVIORS) silently matches nothing, fills every cell
    with 0, and every distance comes out 0.0 -- which looks like perfect
    alignment instead of a bug.
    """
    return (_clean(s).str.lower()
            .map({"neutral": "Neutral", "integrative": "Integrative",
                  "distributive": "Distributive"}))


# =========================================================================
# 1. RAG RETRIEVAL (single_rag architecture only -- build the exemplar bank
#    once per run, then retrieve+format a handful of real human turns per
#    generated turn; see generation/single_agent.py)
# =========================================================================

def build_bank(manual: pd.DataFrame,
               actor_col: str = "ProlificId",
               history_turns: int = 3) -> pd.DataFrame:
    """Each human message becomes a retrievable record tagged with the STATE
    the speaker was in.

    Only state the agent legitimately observes is stored as a key. The Behavior
    label is stored as _behavior (underscore = held out) so you can LOG what was
    retrieved without ever ranking on it.
    """
    m = manual.copy()
    m["Phase"] = _clean(m["Phase"])
    m["Behavior"] = _canon_behavior(m["Behavior"])
    m = m.sort_values(["GameId", "Round"]).reset_index(drop=True)

    rows = []
    for gid, g in m.groupby("GameId"):
        g = g.sort_values("Round").reset_index(drop=True)
        for i, row in g.iterrows():
            actor = row[actor_col]
            past = g.iloc[:i]
            own = past[past[actor_col] == actor]
            partner = past[past[actor_col] != actor]
            rows.append({
                "game_id": gid,
                "actor": actor,
                "round": int(row["Round"]),
                "phase": row["Phase"],
                "role": str(row.get("Role", row.get("Sender", ""))),
                "prev_own": own["Behavior"].iloc[-1] if len(own) else None,
                "prev_partner": partner["Behavior"].iloc[-1] if len(partner) else None,
                "history": "\n".join(past.iloc[-history_turns:]["Text"].astype(str)),
                "text": str(row["Text"]),
                "_behavior": row["Behavior"],          # LOG ONLY
            })
    return pd.DataFrame(rows)


def two_fold_split(manual: pd.DataFrame, rng=RNG) -> dict:
    """Leakage guard. M1 retrieves ONLY from H2 and is compared to H1, and
    vice versa. Both samples stay the same n so the null and the observed
    distance are size-matched.

    Without this the model has seen its own comparison target.
    """
    g = rng.permutation(manual["GameId"].unique())
    h = len(g) // 2
    return {"H1": g[:h], "H2": g[h:2 * h],
            "M1_pool": g[h:2 * h],   # M1 retrieves from H2
            "M2_pool": g[:h]}        # M2 retrieves from H1


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def retrieve(bank: pd.DataFrame,
             query: dict,
             pool_games: Sequence,
             k: int = 5,
             mode: str = "matched",
             embed_fn=None,
             rng=RNG) -> pd.DataFrame:
    """mode: "matched" (state-matched retrieval) | "random" (random baseline)

    Filters are applied in order of importance and RELAXED GRACEFULLY: a filter
    only sticks if it leaves >= k candidates. At this scale a strict conjunctive
    filter (role AND phase AND round AND prev_behavior) will often return 0.

    No weight vector by design. Tuning 6-8 weights on a few dozen dyads is
    guaranteed overfitting and adds researcher degrees of freedom a reviewer
    will flag.
    """
    cand = bank[bank.game_id.isin(pool_games)]
    if len(cand) == 0:
        return cand

    if mode == "random":
        return cand.sample(min(k, len(cand)), random_state=int(rng.integers(1e9)))

    for col, val in [("phase", query.get("phase")),
                     ("role", query.get("role")),
                     ("prev_own", query.get("prev_own"))]:
        if val is None:
            continue
        f = cand[cand[col] == val]
        if len(f) >= k:
            cand = f

    f = cand[(cand["round"] - query["round"]).abs() <= 1]
    if len(f) >= k:
        cand = f

    if embed_fn is not None and query.get("history"):
        q = embed_fn([query["history"]])[0]
        embs = embed_fn(cand["history"].tolist())
        sims = np.array([_cos(q, e) for e in embs])
        return cand.iloc[np.argsort(-sims)[:k]]

    return cand.sample(min(k, len(cand)), random_state=int(rng.integers(1e9)))


def format_exemplars(ex: pd.DataFrame) -> str:
    if len(ex) == 0:
        return ""
    lines = []
    for _, r in ex.iterrows():
        tag = ("after their outside option changed" if r["phase"] == POST
               else "before any change to their outside option")
        lines.append(f'  [Round {r["round"]}, {tag}]\n  "{r["text"]}"')
    return ("\nHow real participants wrote at similar moments:\n\n"
            + "\n\n".join(lines)
            + "\n\nThese show tone and length only. Do NOT reuse their wording. "
              "Write your own message about your own situation.\n")


def copying_rate(logs, n: int = 8) -> float:
    """Share of generated turns containing an 8-gram from their OWN exemplars.

    Above ~5% you are measuring your retrieval system, not model behavior.
    Required table row; first thing a reviewer asks about a retrieval method.
    """
    def grams(t):
        w = re.findall(r"\w+", str(t).lower())
        return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}
    hits = 0
    for r in logs:
        src = set().union(*[grams(e) for e in r["exemplars"]]) if r["exemplars"] else set()
        if grams(r["text"]) & src:
            hits += 1
    return hits / max(1, len(logs))


def retrieval_diagnostic(logs, model_manual: pd.DataFrame) -> pd.DataFrame:
    """What was RETRIEVED vs what was GENERATED.

    If exemplars were 30% distributive and the model produced 55%, you have
    isolated a model-side amplification effect. Nobody reports this.
    """
    ret = [b for r in logs for b in r["retrieved_behaviors"]]
    rv = pd.Series(ret).value_counts(normalize=True).reindex(BEHAVIORS).fillna(0)
    gv = (_canon_behavior(model_manual["Behavior"]).value_counts(normalize=True)
          .reindex(BEHAVIORS).fillna(0))
    return pd.DataFrame({"retrieved": rv, "generated": gv,
                         "amplification": gv - rv}).round(4)


# =========================================================================
# 2. GENERATION OUTPUT SHAPING (generic -- no RAG involved)
# =========================================================================

def to_manual(dialogues, batna_round: int = 3) -> pd.DataFrame:
    """Convert a list of simulated {"game_id", "turns": [...]} dialogues into
    the ds.manual schema (GameId/ProlificId/Role/Round/Phase/Text/Behavior,
    Behavior left NaN -- that's the "unlabeled" part; the classifier fills it
    in later). Used by every generator in this repo (step2_generate.py,
    generate_two_agents.py, smoke_test_two_agents.py), RAG or not.
    """
    rows = []
    for d in dialogues:
        for t in d["turns"]:
            rows.append({"GameId": d["game_id"], "ProlificId": t["actor_id"],
                         "Role": t["role"], "Round": t["round"],
                         "Phase": t["phase"], "Text": t["text"],
                         "Behavior": np.nan})
    return pd.DataFrame(rows)


# =========================================================================
# 3. METRICS
# =========================================================================

def behavior_mix(manual: pd.DataFrame, actor_col="ProlificId",
                 phase: Optional[str] = None) -> np.ndarray:
    """3-D behavior mix: [Neutral, Integrative, Distributive].

    Per-actor rates averaged, so long dyads don't dominate.
    """
    m = manual.copy()
    m["Phase"] = _clean(m["Phase"]); m["Behavior"] = _canon_behavior(m["Behavior"])
    if phase:
        m = m[m["Phase"] == phase]
    per = (m.groupby(["GameId", actor_col])["Behavior"]
             .value_counts(normalize=True).unstack(fill_value=0)
             .reindex(columns=BEHAVIORS, fill_value=0))
    if len(per) == 0:
        # No actor has ANY turn in this phase -- e.g. a small resample that happened
        # to draw only dyads with no Post-BATNA turns. per.mean(axis=0) on an empty
        # frame silently returns [nan, nan, nan] (mean of nothing), which sails past
        # the all-zero check below (nan is never "close to" 0) and then poisons every
        # downstream median/percentile it touches. Raise instead -- two_distance_test
        # catches this and skips the resample rather than let it produce a fake result.
        raise ValueError(
            f"behavior_mix: no rows for phase={phase!r} in this subset -- "
            "likely a too-small/unlucky resample, not a real corpus property.")
    v = per.mean(axis=0).values
    if np.allclose(v, 0):
        raise ValueError(
            "behavior_mix is all zeros -- no rows matched "
            f"{BEHAVIORS}. Check your Behavior label spelling/case, and that "
            f"Phase values are exactly {PRE!r}/{POST!r}.")
    return v


def bootstrap_mix_ci(manual: pd.DataFrame, actor_col="ProlificId",
                     phase: Optional[str] = None, B: int = 500, rng=RNG) -> np.ndarray:
    """95% percentile-bootstrap CI on behavior_mix's 3 proportions, resampling DYADS
    (GameId) with replacement -- a single-sample bootstrap on one side (human or model)
    at a time, unlike two_distance_test's two-sample resampling.

    Returns a (3, 2) array: row i is [lo, hi] for BEHAVIORS[i]. A row is [nan, nan] if
    fewer than half the B draws produced a valid mix (e.g. this phase has too few dyads
    with any turns in it) -- same empty-resample guard as two_distance_test/behavior_mix.
    """
    games = manual["GameId"].unique()
    by_game = {gid: g for gid, g in manual.groupby("GameId")}
    draws = []
    for _ in range(B):
        g = rng.choice(games, len(games), replace=True)
        sub = pd.concat([by_game[x] for x in g], ignore_index=True)
        try:
            draws.append(behavior_mix(sub, actor_col=actor_col, phase=phase))
        except ValueError:
            continue
    if len(draws) < B // 2:
        return np.full((3, 2), np.nan)
    arr = np.array(draws)
    return np.stack([np.percentile(arr, 2.5, axis=0), np.percentile(arr, 97.5, axis=0)], axis=1)


def phase_profile(manual: pd.DataFrame, actor_col="ProlificId") -> np.ndarray:
    """6-D: the 3-way mix in Pre-BATNA concatenated with Post-BATNA.

    WHY THIS EXISTS
    ---------------
    Dropping the 13 subtypes leaves a 3-D vector on a 2-simplex, where L2,
    cosine and JSD all measure nearly the same thing -- you effectively have
    ONE number, and cosine is close to degenerate (all vectors sit in the
    positive orthant, so angles are tiny and uninformative).

    Splitting by phase restores independent dimensions without needing
    subtypes: a model can match the overall mix while getting both phases
    wrong in opposite directions, and only this vector catches that.
    Use it as the PRIMARY static measure.
    """
    return np.concatenate([behavior_mix(manual, actor_col, PRE),
                           behavior_mix(manual, actor_col, POST)])


def length_vector(manual: pd.DataFrame) -> np.ndarray:
    """2-D: [mean words per message, mean turns per dyad], scaled.

    Genuinely independent of the behavior mix, and the cheapest place to catch
    a large gap -- LLMs are typically 3-5x more verbose than real chat users.
    """
    w = manual["Text"].astype(str).str.split().str.len()
    return np.array([w.mean() / 50.0,
                     manual.groupby("GameId").size().mean() / 12.0])


def dist(a, b, kind="L2") -> float:
    if kind == "L2":
        return float(np.linalg.norm(a - b, 2))
    if kind == "L1":
        return float(np.abs(a - b).sum())
    if kind == "cosine":
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(1 - a @ b / (na * nb)) if na and nb else 1.0
    if kind == "jsd":
        p, q = a / a.sum(), b / b.sum()
        mid = 0.5 * (p + q)
        kl = lambda x, y: float((x[x > 0] * np.log2(x[x > 0] / y[x > 0])).sum())
        return 0.5 * kl(p, mid) + 0.5 * kl(q, mid)
    raise ValueError(kind)


# =========================================================================
# 4. THE TEST  (size-matched resampling on BOTH sides)
# =========================================================================

def two_distance_test(human: pd.DataFrame,
                      model: pd.DataFrame,
                      vec_fn=phase_profile,
                      kind: str = "L2",
                      B: int = 1000,
                      n_per_sample: Optional[int] = None,
                      rng=RNG) -> dict:
    """Is dist(humans, model) smaller than dist(humans, humans)?

    BOTH sides are n_per_sample dyads. If you compared 36 humans to 125 model
    dyads, the model side would carry less sampling noise, the distance would
    shrink for purely statistical reasons, and everything would look aligned.

    ratio 1.0 = model is exactly as far from humans as humans are from each other.
    LARGE p_perm = not distinguishable.
    """
    hg = human["GameId"].unique()
    mg = model["GameId"].unique()
    # Clamped to len(hg)//2 no matter what's requested: A=human[sh[:n]] and
    # Bh=human[sh[n:2n]] both come out of the SAME hg pool, so asking for more than
    # half of it starves Bh (e.g. n=47 on a 50-dyad pool leaves Bh only 3 dyads --
    # a small, unlucky draw away from an empty-phase subset and a poisoned result).
    n = min(n_per_sample or len(hg) // 2, len(hg) // 2)

    d_HH, d_HM = [], []
    skipped_hh = skipped_hm = 0
    for _ in range(B):
        sh = rng.permutation(hg)
        A = human[human.GameId.isin(sh[:n])]
        Bh = human[human.GameId.isin(sh[n:2 * n])]

        ms = rng.choice(mg, min(n, len(mg)), replace=False)
        M = model[model.GameId.isin(ms)]

        # vec_fn(A) is shared by both comparisons -- computed once, and if IT raises,
        # both d_HH and d_HM lose this draw together. But d_HH and d_HM are otherwise
        # independent (Bh vs M can each fail on their own), so they're appended in
        # separate try blocks: one failing must not silently drop a value the other
        # side DID successfully compute, which would desync the two arrays' lengths
        # against what "skipped" claims and make each other's B-based thresholds wrong.
        try:
            vA = vec_fn(A)
        except ValueError:
            skipped_hh += 1
            skipped_hm += 1
            continue

        try:
            d_HH.append(dist(vA, vec_fn(Bh), kind))
        except ValueError:
            skipped_hh += 1
        try:
            d_HM.append(dist(vA, vec_fn(M), kind))
        except ValueError:
            skipped_hm += 1

    if skipped_hh or skipped_hm:
        print(f"  [two_distance_test] skipped {skipped_hh}/{B} d_HH and {skipped_hm}/{B} "
              f"d_HM resamples with an empty phase subset (n_per_sample={n})")
    if len(d_HH) < B // 2 or len(d_HM) < B // 2:
        return {
            "metric": kind, "n_per_sample": n, "d_HH_median": np.nan, "d_HH_p95": np.nan,
            "d_HM_mean": np.nan, "d_HM_ci": (np.nan, np.nan), "ratio": np.nan,
            "ratio_ci": (np.nan, np.nan),
            "p_perm": np.nan, "aligned": None,
            "note": f"undefined: only {len(d_HH)}/{B} resamples valid at n_per_sample={n} "
                    "-- too few human/model dyads have data in every phase this metric needs.",
        }

    d_HH, d_HM = np.array(d_HH), np.array(d_HM)
    obs = float(d_HM.mean())
    med = float(np.median(d_HH))
    d_HM_ci = (float(np.percentile(d_HM, 2.5)), float(np.percentile(d_HM, 97.5)))
    # ratio_ci: d_HM's own bootstrap interval rescaled by the fixed d_HH_median point
    # estimate. This propagates the model-distance side's resampling uncertainty only --
    # d_HH_median itself is treated as known, which is reasonable given it is already a
    # median over B=2000 draws (comparatively low-variance next to the single d_HM_mean
    # this ratio is built from), but it is NOT a full two-sided interval the way
    # step6_paired.ratio_ci's nested bootstrap is. State it as such wherever it's reported.
    ratio_ci = (d_HM_ci[0] / med, d_HM_ci[1] / med) if med else (np.nan, np.nan)
    return {
        "metric": kind,
        "n_per_sample": n,
        "d_HH_median": med,
        "d_HH_p95": float(np.percentile(d_HH, 95)),
        "d_HM_mean": obs,
        "d_HM_ci": d_HM_ci,
        "ratio": obs / med if med else np.nan,
        "ratio_ci": ratio_ci,
        "p_perm": float((d_HH >= obs).mean()),
        "aligned": float((d_HH >= obs).mean()) > 0.05,
    }


def calibration_check(human: pd.DataFrame, vec_fn=phase_profile,
                      kind="L2", B=1000) -> dict:
    """Human vs human. MUST give ratio ~1.0 and p ~0.5.
    If it doesn't, your sizing is broken and every other number is meaningless.
    Run this FIRST. Costs nothing — no API calls."""
    g = RNG.permutation(human["GameId"].unique())
    h = len(g) // 2
    return two_distance_test(human[human.GameId.isin(g[:h])],
                             human[human.GameId.isin(g[h:2 * h])],
                             vec_fn, kind, B=B, n_per_sample=h // 2)


def length_stats(manual: pd.DataFrame) -> dict:
    w = manual["Text"].astype(str).str.split().str.len()
    return {"mean_words": float(w.mean()), "median_words": float(w.median()),
            "turns_per_dyad": float(manual.groupby("GameId").size().mean())}


# =========================================================================
# 5. REPORT
# =========================================================================

def full_report(human: pd.DataFrame, conditions: dict, B: int = 1000) -> pd.DataFrame:
    """conditions: {"run": df, ...} -- one or more annotated corpora to compare
    against `human`, each scored the same way (report_replay.py has a richer,
    replay-specific version of this same idea -- this one is step3_classify.py's).

    PRIMARY   phase_profile_L2  (6-D, pre+post mix) -- report this as the headline
    SECONDARY mix_JSD, mix_pre_JSD, mix_post_JSD, length_L2

    Note: with only 3 behavior categories, mix_JSD and mix_L2 are near-redundant
    (both live on a 2-simplex). Report ONE of them, not both, and do not treat
    them as independent tests.
    """
    rows = []
    for name, M in conditions.items():
        for label, vf, kind in [
            ("phase_profile_L2", phase_profile, "L2"),
            ("mix_JSD",          behavior_mix,  "jsd"),
            ("mix_pre_JSD",      lambda m, **kw: behavior_mix(m, phase=PRE, **kw),  "jsd"),
            ("mix_post_JSD",     lambda m, **kw: behavior_mix(m, phase=POST, **kw), "jsd"),
            ("length_L2",        lambda m, **kw: length_vector(m),                  "L2"),
        ]:
            r = two_distance_test(human, M, vf, kind, B=B)
            r.update({"condition": name, "measure": label})
            rows.append(r)
    df = pd.DataFrame(rows)
    return df[["condition", "measure", "d_HH_median", "d_HH_p95",
               "d_HM_mean", "ratio", "p_perm", "aligned", "n_per_sample"]]


USAGE = """
human = ds.manual.copy()

# --- 0. CALIBRATION FIRST. No API calls. Must be ratio ~1.0, p ~0.5 ----------
print(ra.calibration_check(human))   # uses phase_profile by default

# --- 1. generate (see step2_generate.py / step5_replay.py) ------------------
#     No retrieval, no human examples in the prompt -- just generate.

# --- 2. annotate both corpora with YOUR validated coder, BLINDED ------------
#     Auto-code the HUMAN side too, so both sides use the same instrument
#     (classify_replay.py does this for the replay pipeline).

# --- 3. the result ------------------------------------------------------------
print(ra.full_report(human_auto, {"run": M_run}))
"""

if __name__ == "__main__":
    print(__doc__)
    print(USAGE)
