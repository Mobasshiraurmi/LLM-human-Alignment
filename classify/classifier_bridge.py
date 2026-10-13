"""
classifier_bridge.py — wire classify_gpt_batch_final_akash.py into rag_align.py
==============================================================================

Your v6 classifier is already the right tool. It is ablation-refined, uses the
Batch API (50% cheaper, async), carries 5 turns of context, runs at temperature
0, and scores kappa against gold. Do not rebuild it. This file only translates
schemas in both directions.

FLOW
    sim  --sim_to_classifier_csv-->  input CSV
        python classify_gpt_batch_final_akash.py --submit
        python classify_gpt_batch_final_akash.py --check     (repeat)
        python classify_gpt_batch_final_akash.py --parse
    results CSV  --classifier_to_manual-->   labeled df for ra.full_report()

WHAT THE CLASSIFIER EXPECTS
    GameId, Sender, Text, Phase        (Phase optional; it is injected as a hint)

WHAT rag_align PRODUCES
    GameId, ProlificId, Role, Round, Phase, Text, Behavior

So the only real work is Role -> Sender, plus keeping row order stable, because
build_user_message() walks backwards by POSITIONAL index within the same GameId.

MEASUREMENT-INVARIANCE RULE
    Run the SAME classifier version, SAME --scenario value (or none), and SAME
    N_CONTEXT on human and LLM corpora. If humans are hand-coded and models are
    machine-coded, any difference you find is confounded with the coder. You
    already have machine labels on the human side (results_gpt_batch_v6.csv) --
    use those for the comparison and keep the hand labels for your own §4 results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BEHAVIORS = ["Neutral", "Integrative", "Distributive"]

# Mirrors SUBTYPE_TO_BEHAVIOR in the classifier. Note the double-spaced
# "Relationship  building  statement" -- that CSV/model mismatch is the bug the
# v6 scoring fix addressed, so normalize whitespace before mapping.
SUBTYPE_TO_BEHAVIOR = {
    "Neutral": "Neutral",
    "Agreement / finalize remaining issue": "Neutral",
    "Ask(open-ended) questions": "Integrative",
    "Active listening": "Integrative",
    "Share Information": "Integrative",
    "Make integrative(multi-issue) offers": "Integrative",
    "Collaborative Statement": "Integrative",
    "Progress Seeking Statement": "Integrative",
    "Relationship building statement": "Integrative",
    "Ask positionbased question": "Distributive",
    "Discuss one issue at a time": "Distributive",
    "Make single issue offers": "Distributive",
    "Substantiate position or refer to bottom line": "Distributive",
    "Use of force/misrepresentation": "Distributive",
}


def _norm(s):
    return str(s).replace("\u00a0", " ").split() and " ".join(str(s).split()) or ""


# ---------------------------------------------------------------------------
# OUT: simulated dialogues -> classifier input CSV
# ---------------------------------------------------------------------------

def sim_to_classifier_csv(sim_manual: pd.DataFrame, path: str,
                          sort_cols: tuple = ("GameId", "Round")) -> pd.DataFrame:
    """rag_align schema -> the columns classify_gpt_batch expects.

    Sort order matters: build_user_message() collects context by walking
    positional indices backwards while GameId matches, so turns must sit in
    conversational order within each dyad. `["GameId","Round"]` is fine when
    at most one row shares a Round; the turn-grouped two-agent replay passes
    `["GameId","turn_idx","message_index"]` instead, since several turns (and
    several messages within one turn) can share a Round -- Round alone can't
    order them correctly.
    """
    df = sim_manual.copy()
    df["Sender"] = df["Role"] if "Role" in df.columns else df["ProlificId"]
    keep = ["GameId", "Sender", "Text", "Phase", "ProlificId", "Round"]
    keep += [c for c in sort_cols if c not in keep and c in df.columns]
    df = df.sort_values(list(sort_cols)).reset_index(drop=True)[keep]
    df.to_csv(path, index=False)
    print(f"{len(df)} turns -> {path}  ({df.GameId.nunique()} dyads)")
    print("Now run:")
    print(f"  # edit INPUT_CSV = '{path}' and OUTPUT_CSV / BATCH_* filenames")
    print("  python classify_gpt_batch_final_akash.py --submit")
    print("  python classify_gpt_batch_final_akash.py --check    # repeat")
    print("  python classify_gpt_batch_final_akash.py --parse")
    return df


# ---------------------------------------------------------------------------
# BACK IN: classifier results -> labeled manual
# ---------------------------------------------------------------------------

def classifier_to_manual(results_csv: str,
                         original: pd.DataFrame,
                         behavior_col: str = "pred_behavior",
                         subtype_col: str = "pred_subtype",
                         sort_cols: tuple = ("GameId", "Round")) -> pd.DataFrame:
    """Attach the classifier's Behavior label back onto the rag_align dataframe.

    PREFERS the classifier's own `pred_behavior` column. Only falls back to
    re-deriving Behavior from `pred_subtype` if that column is absent.

    WHY THIS ORDER MATTERS
    ----------------------
    The classifier emits BOTH a subtype and a behavior, and they do not always
    agree -- re-mapping subtypes can move hundreds of messages between classes.
    Your human corpus is scored on `pred_behavior`, so the model corpora must be
    too. Mixing the two means any human-model difference you report is partly
    the instrument rather than the agents.

    `sort_cols` MUST match whatever `sim_to_classifier_csv` sorted `original` by
    when building the classifier input -- this join is still positional
    (`results_csv` and `original` are independently re-sorted the same way and
    zipped by `.values`), just no longer assuming `["GameId","Round"]` is a
    sufficient key when several rows can share a Round.
    """
    res = pd.read_csv(results_csv)
    if len(res) != len(original):
        raise ValueError(f"Row mismatch: results={len(res)} original={len(original)}. "
                         "Some batch requests failed - reconcile before joining.")

    out = original.sort_values(list(sort_cols)).reset_index(drop=True).copy()

    if behavior_col in res.columns and res[behavior_col].notna().any():
        out["Behavior"] = res[behavior_col].map(_norm).values
        src = behavior_col
    else:
        cand = [c for c in res.columns
                if "subtype" in c.lower() and "gold" not in c.lower()]
        col = subtype_col if subtype_col in res.columns else (cand[0] if cand else None)
        if col is None:
            raise ValueError(f"No behavior or subtype column in {results_csv}: "
                             f"{list(res.columns)}")
        print(f"  ! no {behavior_col} column - re-deriving from {col}. "
              "Make sure the human side is scored the same way.")
        subs = res[col].map(_norm)
        out["BehaveSubtype"] = subs.values
        out["Behavior"] = subs.map(SUBTYPE_TO_BEHAVIOR).values
        src = col

    if subtype_col in res.columns:
        out["BehaveSubtype"] = res[subtype_col].map(_norm).values

    bad = out["Behavior"].isna().sum()
    if bad:
        print(f"  WARNING: {bad} unmapped labels from {src}")
    out = out.dropna(subset=["Behavior"])
    print(f"  scored from {src!r}:")
    print(out["Behavior"].value_counts().to_string())
    return out


# ---------------------------------------------------------------------------
# TURN-GROUPED REPLAY: fold message-level labels down to one per turn
# ---------------------------------------------------------------------------

def fold_messages_to_turns(labeled_messages: pd.DataFrame,
                           turn_keys: tuple = ("GameId", "turn_idx")) -> pd.DataFrame:
    """One row per TURN, using its CLOSING message's Behavior.

    Only for metrics that genuinely need a single paired label per turn (turn-level
    agreement: accuracy, Gwet's AC1, per-class recall) -- taking the last message is
    how the codebase already separates a leading reaction from the substantive move
    (see two_agents.ACTION_TO_BEHAVIOR), generalized to any burst length instead of a
    forced 2-message split.

    Do NOT use this for distributional metrics (behavior_mix, ΔDistributive, style
    vectors, path dependence) -- those read one side's labels by phase/dyad and never
    join human to model, so folding first would silently discard every non-final burst
    message (typically the acknowledgment) from the mix. Use the message-level frame
    directly for those.
    """
    keys = list(turn_keys)
    return (labeled_messages.sort_values(keys + ["message_index"])
            .groupby(keys, as_index=False).last())


# ---------------------------------------------------------------------------
# THE CHECK THAT MATTERS MOST HERE
# ---------------------------------------------------------------------------

def differential_error_check(human_gold: pd.Series,
                             human_pred: pd.Series,
                             llm_gold: pd.Series,
                             llm_pred: pd.Series) -> pd.DataFrame:
    """Is the classifier as accurate on LLM text as on human text?

    WHY THIS IS THE ONE NEW THING YOU NEED
    --------------------------------------
    Your v6 prompt was tuned on HUMAN messages -- the fix list names human
    artifacts like screen observations ("on my screen i see 60k") and typo
    corrections. LLM turns are longer, cleaner, and more explicitly cooperative.
    If the classifier is (say) kappa .82 on human text and .65 on LLM text, then
    part of any human-LLM gap you report is the CODER, not the agents.

    Cost: hand-code ~150 LLM turns once. That is the only new annotation in the
    whole pipeline, and it converts a fatal objection into a reported number.

    A gap under ~.10 is fine to report as comparable. Above that, say so
    explicitly and treat it as a bound on your effect.
    """
    from sklearn.metrics import cohen_kappa_score, f1_score

    rows = []
    for name, g, p in [("human", human_gold, human_pred),
                       ("llm", llm_gold, llm_pred)]:
        g, p = pd.Series(g).map(_norm), pd.Series(p).map(_norm)
        m = g.notna() & p.notna()
        g, p = g[m], p[m]
        rows.append({
            "corpus": name, "n": len(g),
            "accuracy": float((g == p).mean()),
            "kappa": float(cohen_kappa_score(g, p)),
            "f1_distributive": float(f1_score(g == "Distributive",
                                              p == "Distributive",
                                              zero_division=0)),
            "f1_integrative": float(f1_score(g == "Integrative",
                                             p == "Integrative",
                                             zero_division=0)),
            "f1_neutral": float(f1_score(g == "Neutral", p == "Neutral",
                                         zero_division=0)),
        })
    df = pd.DataFrame(rows)
    gap = abs(df.kappa.iloc[0] - df.kappa.iloc[1])
    print(df.round(3).to_string(index=False))
    print(f"\nkappa gap: {gap:.3f}")
    print("  < .10  -> comparable, report both and move on")
    print("  >= .10 -> DIFFERENTIAL ERROR. Report it; it bounds your effect.")
    return df


def cross_coder_agreement(labels_a: pd.Series, labels_b: pd.Series,
                          corpus_name: str = "") -> dict:
    """Reliability WITHOUT gold labels: how much do two automated coders agree?

    Run the v6 classifier twice per corpus under a deliberate variation --
    a different MODEL, or N_CONTEXT=5 vs N_CONTEXT=2 -- then measure agreement
    between the two label sets. Do this on the human corpus AND each LLM corpus.

    The logic: if the coder agrees with itself/another coder at AC1 .85 on human
    text but only .68 on LLM text, it is measurably less reliable on LLM text,
    and part of any human-model gap you report is the instrument. You learn this
    with zero hand coding.

    Reports Gwet's AC1 alongside Cohen's kappa because your classes are skewed
    (~14% Distributive) and kappa deflates under skewed marginals -- the "kappa
    paradox". Li et al. (2025) use AC1 for exactly this reason.
    """
    from sklearn.metrics import cohen_kappa_score

    a, b = pd.Series(labels_a).map(_norm), pd.Series(labels_b).map(_norm)
    m = a.notna() & b.notna() & (a != "") & (b != "")
    a, b = a[m], b[m]
    n = len(a)
    po = float((a == b).mean())

    # Gwet's AC1
    cats = sorted(set(a) | set(b))
    pi = [((a == c).mean() + (b == c).mean()) / 2 for c in cats]
    q = len(cats)
    pe = sum(p * (1 - p) for p in pi) / (q - 1) if q > 1 else 0.0
    ac1 = (po - pe) / (1 - pe) if pe < 1 else np.nan

    out = {"corpus": corpus_name, "n": n, "percent_agreement": po,
           "gwet_ac1": float(ac1),
           "cohens_kappa": float(cohen_kappa_score(a, b)) if a.nunique() > 1 else np.nan}
    print("  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in out.items()))
    return out


def reliability_report(pairs: dict) -> pd.DataFrame:
    """pairs: {"human": (labels_run1, labels_run2), "run": (...), ...}

    Interpretation of the AC1 gap between human and each LLM corpus:
        < .10   comparable reliability -- report and move on
        >= .10  the instrument is less reliable on LLM text. Report it as a
                bound on your effect; do not hide it.
    """
    rows = [cross_coder_agreement(a, b, name) for name, (a, b) in pairs.items()]
    df = pd.DataFrame(rows)
    if "human" in df.corpus.values:
        h = df.loc[df.corpus == "human", "gwet_ac1"].iloc[0]
        df["ac1_gap_vs_human"] = (df.gwet_ac1 - h).abs().round(3)
    print()
    print(df.round(3).to_string(index=False))
    return df


def label_stability_diagnostics(manual: pd.DataFrame) -> dict:
    """Cheap signals that the coder is struggling, no second run required.

    - unmapped: subtypes the model emitted that are not in the codebook
    - degenerate: share of turns in the single most frequent class
    A much higher degenerate rate on LLM text than human text means the coder is
    collapsing toward one label there.
    """
    sub = manual["BehaveSubtype"].map(_norm)
    beh = manual["Behavior"].map(_norm)
    return {
        "n": len(manual),
        "unmapped_rate": float((~sub.isin(SUBTYPE_TO_BEHAVIOR)).mean()),
        "degenerate_rate": float(beh.value_counts(normalize=True).max()),
        "n_distinct_subtypes": int(sub.nunique()),
    }


def reproduction_check(human_hand: pd.DataFrame,
                       human_auto: pd.DataFrame) -> pd.DataFrame:
    """Does machine coding recover your hand-coded human numbers?

    If yes, one sentence kills the measurement-invariance objection:
    'the validated classifier reproduced the hand-coded distribution
    (Distributive .142 vs .139), so human and model corpora are compared
    under a single instrument.'
    """
    def mix(df):
        return (df["Behavior"].map(_norm).value_counts(normalize=True)
                .reindex(BEHAVIORS).fillna(0))
    h, a = mix(human_hand), mix(human_auto)
    out = pd.DataFrame({"hand_coded": h, "auto_coded": a, "diff": a - h}).round(4)
    print(out.to_string())
    return out


RUNBOOK = """
PER-CORPUS RUN

For the replay pipeline (step5_replay.py --agent {single,multi}), all of this
is already wired up in classify_replay.py -- just run
    python classify_replay.py --agent single --submit / --check / --parse
    python classify_replay.py --agent multi  --submit / --check / --parse
and report_replay.py --agent {single,multi} for the analysis. What follows is
the lower-level manual version, for the step2_generate.py (free-running,
"sim/") corpus or any one-off corpus not already covered by classify_replay.py.

1. Export
     import classifier_bridge as cb
     cb.sim_to_classifier_csv(sim, "clf_input.csv")

2. In classify_gpt_batch_final_akash.py, point the filenames at this corpus:
     INPUT_CSV          = "clf_input.csv"
     OUTPUT_CSV         = "results.csv"
     BATCH_INPUT_JSONL  = "batch_in.jsonl"
     BATCH_OUTPUT_JSONL = "batch_out.jsonl"
     BATCH_ID_FILE      = "batch_id.txt"
   Leave MODEL, N_CONTEXT, TEMPERATURE, and the prompt UNTOUCHED.
   Use the same --scenario value (or none) for every corpus including human.

3. Submit, then wait. Batch turnaround is usually well under the 24h window,
   and you are billed at the 50% batch rate.
     python classify_gpt_batch_final_akash.py --submit
     python classify_gpt_batch_final_akash.py --check
     python classify_gpt_batch_final_akash.py --parse

4. Back in
     sim = cb.classifier_to_manual("results.csv", sim)

5. Analysis (unchanged)
     import rag_align as ra
     print(ra.full_report(human_auto, {"run": sim}, B=1000))

NOTES
-----
* Rotate the API key that was hardcoded on line 57, then read it from the
  environment:  OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
* --parse computes kappa against a gold column. Simulated corpora have no gold,
  so it will report nothing there; that is expected, not a failure.
* You already have machine labels on the human side. Use those for the
  human-vs-model comparison and keep the hand labels for your own §4 results.
* NO hand coding required. Instead run each corpus TWICE under one deliberate
  variation (different MODEL, or N_CONTEXT 5 vs 2) and use reliability_report()
  to compare agreement on human vs LLM text. Batch API makes the second pass
  cheap. This is a reliability check, not a validity check -- say so in the
  limitations, and note that the classifier was validated against gold on human
  text only.
"""

if __name__ == "__main__":
    print(__doc__)
    print(RUNBOOK)