"""
step1_calibration.py — run this first. No API key, no ds, no generation.
=======================================================================

Answers one question: at N=36 dyads, can this design detect anything at all?

It splits your HUMAN dyads in half 1000 times and measures how far apart two
groups of real humans are. That distance is the threshold everything else is
judged against. The sanity check is that human-vs-human should give
ratio ~1.0 and p ~0.5 -- if it doesn't, the sizing is broken and every
downstream number is meaningless.

Usage:
    python generation/step1_calibration.py path/to/your_annotated.csv
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stats"))
import rag_align as ra


# --- column name candidates, since your files differ -----------------------
CANDIDATES = {
    "GameId":     ["GameId", "game_id", "DyadId", "Dyad"],
    "ProlificId": ["ProlificId", "Sender", "actor", "ActorId", "Participant"],
    "Round":      ["Round", "round", "Turn", "turn"],
    "Phase":      ["Phase", "phase"],
    "Behavior":   ["Behavior", "behavior", "gold_label", "Gold", "family"],
    "Text":       ["Text", "text", "Message", "message"],
}


def load(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    print(f"loaded {len(raw)} rows")
    print("columns:", list(raw.columns), "\n")

    df, missing = pd.DataFrame(), []
    for want, opts in CANDIDATES.items():
        hit = next((c for c in opts if c in raw.columns), None)
        if hit:
            df[want] = raw[hit]
            if hit != want:
                print(f"  mapped {hit!r} -> {want!r}")
        else:
            missing.append((want, opts))

    if missing:
        print("\nCOULD NOT FIND:")
        for want, opts in missing:
            print(f"  {want}  (looked for: {opts})")
        print("\nEdit CANDIDATES at the top of this file to add your column name.")
        sys.exit(1)

    df["Phase"] = df["Phase"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["Behavior"] = ra._canon_behavior(df["Behavior"])
    bad = df["Behavior"].isna().sum()
    if bad:
        print(f"\n!! {bad} rows have unrecognized Behavior labels; dropping them.")
    df["Round"] = pd.to_numeric(df["Round"], errors="coerce")
    return df.dropna(subset=["Round", "Behavior"])


def main(path: str):
    df = load(path)

    print("\n" + "=" * 62)
    print("DATA")
    print("=" * 62)
    print(f"messages : {len(df)}")
    print(f"dyads    : {df.GameId.nunique()}")
    df["_actor"] = df.GameId.astype(str) + "|" + df.ProlificId.astype(str)
    print(f"actors   : {df._actor.nunique()}   "
          f"({df.ProlificId.nunique()} distinct role labels)")
    print(f"\nBehavior:\n{df.Behavior.value_counts()}")
    print(f"\nPhase:\n{df.Phase.value_counts()}")

    # Phase labels must match what rag_align expects
    if ra.PRE not in set(df.Phase) or ra.POST not in set(df.Phase):
        print(f"\n!! Phase values are {sorted(set(df.Phase))}")
        print(f"!! rag_align expects {ra.PRE!r} and {ra.POST!r}. Remap before continuing.")
        sys.exit(1)

    n_dyads = df.GameId.nunique()
    print("\n" + "=" * 62)
    print(f"CALIBRATION  (n={n_dyads//4} dyads per side)")
    print("=" * 62)

    cal = ra.calibration_check(df, B=1000)
    for k, v in cal.items():
        print(f"  {k:16s} {v}")

    ok = 0.7 < cal["ratio"] < 1.4
    print("\n  " + ("PASS -- sizing is sound, proceed to Step 2"
                    if ok else
                    "FAIL -- ratio far from 1.0. Check dyad counts before spending on generation."))

    # --- how much room does the design actually have? ---------------------
    print("\n" + "=" * 62)
    print("POWER: is the null tight enough to detect anything?")
    print("=" * 62)
    print(f"  null median (typical human-vs-human distance) : {cal['d_HH_median']:.4f}")
    print(f"  null p95    (extreme human-vs-human distance) : {cal['d_HH_p95']:.4f}")
    if cal["d_HH_median"] > 0:
        print(f"  spread ratio p95/median                       : "
              f"{cal['d_HH_p95']/cal['d_HH_median']:.2f}")
    else:
        print("  spread ratio                                  : undefined")
        print("\n  !! A null of exactly 0 means the metric found no variation at all.")
        print("  !! Almost always a label-matching bug, not a real result.")
        sys.exit(1)
    print("\n  A spread ratio above ~3 means the null is wide: 'aligned' will be")
    print("  easy to achieve and a reviewer will say so. Report the null width")
    print("  next to every result rather than waiting to be asked.")

    # --- your frozen human targets ----------------------------------------
    print("\n" + "=" * 62)
    print("FROZEN HUMAN TARGETS  (record these before any model runs)")
    print("=" * 62)
    labels = [f"{p}_{b}" for p in ["pre", "post"] for b in ra.BEHAVIORS]
    prof = ra.phase_profile(df)
    for lab, v in zip(labels, prof):
        print(f"  {lab:20s} {v:.4f}")
    print(f"\n  overall mix (N/I/D) : {ra.behavior_mix(df).round(4)}")
    print(f"  length              : {ra.length_stats(df) if 'Text' in df else 'n/a'}")

    delta = prof[3:] - prof[:3]
    print(f"\n  post - pre shift    : {dict(zip(ra.BEHAVIORS, delta.round(4)))}")
    print("  (sanity: Distributive should be ~ +0.09 per your Table 1)")

    out = pd.DataFrame({"metric": labels, "value": prof})
    out.to_csv("frozen_human_targets.csv", index=False)
    print("\n  saved -> frozen_human_targets.csv")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])