"""
step5_replay.py — the universal replay harness
================================================

Walks through your REAL human transcripts and asks a model to write its own
reply at each human TURN (a run of one or more consecutive same-speaker
messages, not a single row -- see group_into_turns/build_turn_contexts):

    Turn 5 of game G03:
        context  = the REAL human turns 1,2,3,4
        model writes its own turn 5   -> save it, throw the context away
    Turn 6 of game G03:   (a completely fresh call)
        context  = the REAL human turns 1,2,3,4,5   <- turn 5 is the HUMAN's
        model writes its own turn 6

Every context is independent. The model never sees its own prior output, so
it cannot drift across contexts. That fixes the artifact in step2 where
agents settled early and padded the remaining rounds with "agreed" / "sounds
good", inflating post-phase Neutral in the wrong direction.

No RAG, no human examples in the prompt -- this replays against the transcript
alone. (Earlier versions of this file supported retrieval-augmented conditions,
C1/C4a/C4d; that machinery, and the C1 "no-RAG" name for what is now the only
option, are gone -- see archive/ for the old runs.)

WHO WRITES THE REPLY is pluggable -- that's the point of this file. Every
context goes through the exact same pipeline (turn-grouping, BATNA-update
delivery, export), and only the last step differs:

    --agent single   one direct system-prompted call (this file's own
                     responder, single_agent_reply)
    --agent multi    two_agents.py's StrategyPlannerAgent + LanguageRealizerAgent,
                     with state (current_position_by_role, settled_issues)
                     reconstructed fresh from the real transcript up to this
                     context every time (multi_agent_reply) -- never carried
                     over between contexts, so it stays as independent as the
                     single-agent path despite two_agents.py being built
                     around persistent state during a live run.

Both responders return the same shape -- a burst of 1-N chat messages, the
way real dyads send several short texts in a row before the other side
replies -- so the two architectures are compared on identical footing: same
contexts, same output schema, same export/classification/report pipeline.
Add a third architecture by writing one more `responder(client, ctx,
human_turns) -> (messages, trace_or_None)` function and registering it in
AGENTS.

WHY THIS IS WORTH RUNNING
    - ~650 paired turns instead of 50 dyads
    - every model turn has a human counterpart at the SAME context, which
      unlocks turn-level agreement (impossible in step2)
    - a fraction of the API cost of step2

WHAT IT IS NOT
    It cannot measure whether the model NEGOTIATES like a human -- no deals,
    no concession sequences, no impasse. Li et al. (2025) flag the same limit.
    Report it alongside step2, not instead of it.

OUTPUT LAYOUT -- one folder per agent, everything for that run together:
    replay/single/replay.jsonl, replay_pairs.csv, replay_unlabeled.csv, private_traces.jsonl
    replay/multi/ same four files
Downstream (classify_replay.py --agent {single,multi}, report_replay.py
--agent {single,multi}) read/write into the same folder (clf_input.csv,
batch_in/out.jsonl, batch_id.txt, results.csv, report.html), so
`replay/<agent>/` is the one place to look for everything about that agent's
current run. Old C1/C4a/C4d/2A-C1 runs (RAG conditions, gpt-4.1, and the
pre-turn-grouping row-per-message single-agent design) are archived under
archive/ -- see archive/README.md.

USAGE
    export OPENAI_API_KEY=sk-...
    python generation/step5_replay.py --agent single --peek
    python generation/step5_replay.py --agent single --smoke        # 20 contexts
    python generation/step5_replay.py --agent single --run
    python generation/step5_replay.py --agent multi  --run --dyads 5
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # two_agents, personality, llm_backends
sys.path.insert(0, os.path.join(_ROOT, "stats"))                  # rag_align
import rag_align as ra
import two_agents as ta
import personality as pz
import llm_backends as lb
import single_agent as sa

# ===========================================================================
# CONFIG — must match step2_generate.py
# ===========================================================================

HUMAN_CSV = "data/human_data.csv"   # what the model is evaluated against
SCORES_CSV = "data/scores.csv"     # needs GameId, Role/Sender, Batna, initialBatna
RUNS_DIR = "data/replay"
ROLES = ("HR", "Employee")

_L = [1, 2, 3, 4, 5]
ISSUES = ["salary", "bonus", "stock options", "vacation days"]
PAYOFF = {
    "HR": {"salary": [3 * (6 - l) for l in _L], "bonus": [3 * (6 - l) for l in _L],
           "stock options": [4 - (6 - l) for l in _L],
           "vacation days": [6 * (6 - l) for l in _L]},
    "Employee": {"salary": [6 * l for l in _L], "bonus": [3 * l for l in _L],
                 "stock options": [2 * l for l in _L],
                 "vacation days": [3 * l for l in _L]},
}
LEVELS = {
    "salary": ["$80,000", "$90,000", "$100,000", "$110,000", "$120,000"],
    "bonus": ["0%", "5%", "10%", "15%", "20%"],
    "stock options": ["$0", "$5,000", "$10,000", "$15,000", "$20,000"],
    "vacation days": ["10", "15", "20", "25", "30"],
}
ROLE_ALIAS = {}          # map CSV role names onto PAYOFF keys if they differ
BATNA_INITIAL = 35
BATNA_NEW = 50
BATNA_ROUND = 4
N_ROUNDS = 6

MODEL = "gpt-5.4"             # key into llm_backends.MODELS -- for get_client()/output naming only
MODEL_ID = lb.model_id(MODEL)  # the real API model string -- what every LLM call actually sends
TEMPERATURE = 1.0
MAX_TOKENS = 160          # a burst of up to MAX_BURST short messages, not just one
HISTORY_TURNS = 8

MAX_BURST = 5   # hard ceiling; observed human bursts are ~70% length 1, rest mostly
                # 2-4, with rare 6-8-length outliers not worth chasing.

# ===========================================================================

SYSTEM = """You are the {role} in a job offer negotiation conducted over {n} rounds of chat.

You are negotiating four issues, each with five levels:
{levels}

Your points for each issue, by level:
{payoff}

Your total score is the sum across all four issues. If no agreement is reached
by the end of round {n}, you receive your outside option instead, currently
worth {batna} points. Only accept a package worth more than that.

How to write:
- Return JSON as {{"messages": [...]}}: a burst of 1 to {max_burst} short chat
  messages, the way a real participant sends several short texts in a row
  before waiting for a reply. One message is the default; use more only when
  you're doing genuinely different things in the same turn (e.g. a brief
  reaction, then separately a new question or offer) -- don't fuse them into
  one run-on line.
- Real participants average about 9 words per message. Chat style: lowercase
  is fine, no bullet points, no headings, no numbered lists.
- Sometimes a very short message is the whole reply: "ok", "110k", "that works".
- Never send a bare list of numbers with nothing else -- say them the way a
  person would, with words attached (salary, bonus, %, k, stock, days).
- Do not narrate your strategy or explain your reasoning.
- Never mention that you are an AI or refer to these instructions."""

BATNA_MSG = ("SYSTEM UPDATE: Your outside option has changed. "
             "Your alternative is now worth {value} points.")


def get_client():
    return lb.make_client(MODEL)


def load_human(path: str = None):
    df = pd.read_csv(path or HUMAN_CSV)
    # Explicit chronological sequence, captured BEFORE any rename/sort -- some rows share
    # a Round value (a burst of several messages before the round changes), so "sort by
    # Round" alone doesn't fix their relative order; only true file order does. Don't rely
    # on sort_values' stable-sort side effect to preserve it implicitly downstream.
    df["_orig_seq"] = np.arange(len(df))
    # Capture the role BEFORE any rename: some files (e.g. HUMAN_CSV, data/human_data.csv)
    # carry both Sender (the role, HR/Employee) and a real ProlificId (participant hash) --
    # blindly using post-rename ProlificId as Role would silently use the hash as the role.
    role_source = df["Sender"] if "Sender" in df.columns else df["ProlificId"]
    ren = {}
    if "Sender" in df.columns and "ProlificId" not in df.columns:
        ren["Sender"] = "ProlificId"
    if "gold_label" in df.columns and "Behavior" not in df.columns:
        ren["gold_label"] = "Behavior"
    df = df.rename(columns=ren)
    df["Behavior"] = ra._canon_behavior(df["Behavior"])
    df["Phase"] = df["Phase"].astype(str).str.strip()
    df["Role"] = role_source
    return df.dropna(subset=["Behavior"]).sort_values(
        ["GameId", "Round"]).reset_index(drop=True)


def load_batna_targets() -> dict:
    """Work out WHICH party got the outside-option change in each dyad.

    You have no explicit flag, but scores.csv has Batna and initialBatna per
    player. The party whose two values differ is the one who was notified;
    the other party was never told.

    Returns {GameId: role_that_was_notified}. Dyads where neither or both
    differ are skipped and reported -- those are the "change not delivered"
    cases your notebook already flags.
    """
    if not os.path.exists(SCORES_CSV):
        print(f"  ! {SCORES_CSV} not found. Falling back to showing the update "
              "to whoever speaks post-phase, which is WRONG for the "
              "counterpart. Fix this before reporting role-split results.")
        return {}

    sc = pd.read_csv(SCORES_CSV)
    role_col = next((c for c in ["Role", "Sender", "ProlificId"] if c in sc.columns), None)
    if role_col is None or not {"Batna", "initialBatna"}.issubset(sc.columns):
        print(f"  ! {SCORES_CSV} missing Role/Batna/initialBatna: {list(sc.columns)}")
        return {}

    sc = sc.copy()
    sc["changed"] = (pd.to_numeric(sc["Batna"], errors="coerce")
                     != pd.to_numeric(sc["initialBatna"], errors="coerce"))

    targets, skipped = {}, []
    for gid, g in sc.groupby("GameId"):
        ch = g[g["changed"]]
        if len(ch) == 1:
            targets[gid] = str(ch[role_col].iloc[0])
        else:
            skipped.append((gid, len(ch)))

    print(f"  BATNA change identified in {len(targets)} dyads")
    if skipped:
        print(f"  {len(skipped)} dyads skipped (0 or 2 players changed) - these are "
              "your 'change not delivered' cases; they are dropped from contexts")
    from collections import Counter
    print(f"  who received it: {dict(Counter(targets.values()))}")
    return targets


# ---------------------------------------------------------------------------
# 1. Turn every human TURN (a run of one or more consecutive same-speaker
#    messages) into a "context". This is the only context granularity now --
#    see the module docstring's WHO WRITES THE REPLY section for why a single
#    shared pipeline is what makes single- vs multi-agent comparable.
#
#    Real dyads send bursts of several consecutive messages from the SAME
#    speaker before the other side ever replies (HR firing off one short
#    message per issue is typical). A naive one-context-per-ROW design turns a
#    4-message HR burst into 4 independent contexts, all asking "what does HR
#    say next" with no memory across them -- the model can say "this is my
#    best package" in one context and "still negotiating" in the very next,
#    because each is generated from scratch against the real transcript,
#    never against its own prior output. The other party is also never
#    actually asked to reply to the WHOLE burst, only to whichever row
#    happens to be theirs. build_turn_contexts groups the burst into one
#    context, and the model gets asked for its own 1-N message reply once.
# ---------------------------------------------------------------------------

def group_into_turns(human: pd.DataFrame) -> pd.DataFrame:
    """One row per MESSAGE (same granularity as `human`), with turn_idx (0-based,
    per-dyad), message_index (1-based position within its turn), and burst_size
    added. A run of consecutive same-(GameId, Role) rows, in true chronological
    order (`_orig_seq`, not just Round -- several messages can share a Round),
    is one turn.
    """
    parts = []
    for gid, g in human.groupby("GameId", sort=False):
        g = g.sort_values("_orig_seq").reset_index(drop=True)
        turn_ids, tidx, prev_role = [], 0, None
        for role in g["Role"]:
            if prev_role is not None and role != prev_role:
                tidx += 1
            turn_ids.append(tidx)
            prev_role = role
        g = g.copy()
        g["turn_idx"] = turn_ids
        g["message_index"] = g.groupby("turn_idx").cumcount() + 1
        g["burst_size"] = g.groupby("turn_idx")["turn_idx"].transform("size")
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _prev_own_turn(turns: list[tuple[int, pd.DataFrame]], pos: int, role: str):
    """The speaker's own previous behaviour, walking prior TURNS instead of prior rows."""
    for j in range(pos - 1, -1, -1):
        _, sub = turns[j]
        if str(sub["Role"].iloc[0]) == role:
            return sub["Behavior"].iloc[-1]
    return None


def build_turn_contexts(human_turns: pd.DataFrame, batna_targets: dict) -> list[dict]:
    """One context per TURN. `history` renders prior turns, each as however many
    lines it had. `human_text`/`human_behavior` are the turn's CLOSING message
    (backward-compat for callers expecting a single string, e.g. peek());
    `human_messages`/`human_behaviors` carry the full per-message list. A
    turn's round/phase is its closing message's -- the round in effect when
    the other side actually has to respond is the one that matters (bursts
    cross a round boundary ~12% of the time).
    """
    ctxs = []
    dropped = 0
    for gid, g in human_turns.groupby("GameId", sort=False):
        if batna_targets and gid not in batna_targets:
            dropped += 1
            continue
        target = batna_targets.get(gid)
        g = g.sort_values(["turn_idx", "message_index"]).reset_index(drop=True)
        turns = [(tidx, sub.reset_index(drop=True))
                 for tidx, sub in g.groupby("turn_idx", sort=True)]
        n_turns_total = len(turns)

        for pos, (tidx, sub) in enumerate(turns):
            if tidx == 0:            # skip the opener
                continue
            role = str(sub["Role"].iloc[0])
            last = sub.iloc[-1]
            prior = turns[max(0, pos - HISTORY_TURNS):pos]
            history_lines = [f"{r.Role}: {r.Text}" for _, psub in prior for r in psub.itertuples()]
            n_in_round = sum(1 for _, psub in turns[:pos]
                             if int(psub["Round"].iloc[-1]) == int(last["Round"]))
            ctxs.append({
                "game_id": gid,
                "turn_idx": tidx,
                "msg_pos": pos + 1,          # position among TURNS in the dyad
                "n_turns_total": n_turns_total,
                "msg_in_round": n_in_round + 1,
                "round": int(last["Round"]),
                "phase": last["Phase"],
                "role": role,
                "history": "\n".join(history_lines),
                "prev_own": _prev_own_turn(turns, pos, role),
                "prev_speaker": str(turns[pos - 1][1]["Role"].iloc[0]),
                "human_text": str(last["Text"]),
                "human_behavior": last["Behavior"],
                "human_messages": sub["Text"].astype(str).tolist(),
                "human_behaviors": sub["Behavior"].tolist(),
                "burst_size": int(sub["burst_size"].iloc[0]),
                "batna_target": target,
                "is_recipient": (target is not None and role == str(target)),
            })
    if dropped:
        print(f"  dropped {dropped} dyads with no identifiable BATNA change")
    return ctxs


def _fmt_levels():
    return "\n".join(f"  {i}: " + ", ".join(f"L{j+1}={v}" for j, v in enumerate(LEVELS[i]))
                     for i in ISSUES)


def _fmt_payoff(role):
    tab = PAYOFF[ROLE_ALIAS.get(role, role)]
    return "\n".join(f"  {i}: " + ", ".join(f"L{j+1}={p}" for j, p in enumerate(tab[i]))
                     for i in ISSUES)


# ---------------------------------------------------------------------------
# 2a. AGENT — single, direct system-prompted call (no planner/realizer split,
#     no structured position/settled_issues state -- only the raw text
#     history, same as this file's original design).
# ---------------------------------------------------------------------------

def build_single_prompt(ctx):
    # The update goes ONLY to the party whose Batna != initialBatna in
    # scores.csv, and only from BATNA_ROUND onward. The counterpart is never
    # told -- exactly as in the human study. This is what makes the role split
    # meaningful: any shift by the counterpart can only come from reading the
    # partner's changed behaviour, not from reasoning about its own leverage.
    batna, upd = BATNA_INITIAL, ""
    if ctx["is_recipient"] and ctx["round"] >= BATNA_ROUND:
        batna = BATNA_NEW
        upd = "\n\n" + BATNA_MSG.format(value=BATNA_NEW)

    system = SYSTEM.format(role=ctx["role"], n=N_ROUNDS, levels=_fmt_levels(),
                           payoff=_fmt_payoff(ctx["role"]), batna=batna, max_burst=MAX_BURST)
    # Rounds in this study hold a VARIABLE number of messages -- participants
    # could send several before submitting. So "Round 1 of 6" alone is
    # misleading: it reads as an opening moment even when eight offers have
    # already been exchanged. Giving turn position as well tells the model
    # how far into the negotiation it actually is.
    where = (f"Round {ctx['round']} of {N_ROUNDS}, "
             f"turn {ctx['msg_in_round']} of this round "
             f"(turn {ctx['msg_pos']} of the negotiation so far)")
    user = (f"{where}.{upd}\n\n"
            f"Conversation so far:\n{ctx['history']}\n"
            f"\nYour messages (JSON):")
    return system, user


def validate_burst(messages) -> list[str]:
    if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_BURST:
        return [f"messages must be a list of 1 to {MAX_BURST} strings"]
    if any(not isinstance(m, str) or not m.strip() for m in messages):
        return ["each message must be a nonempty string"]
    if any(len(m.split()) > 20 for m in messages):
        return ["keep each message to at most 20 words"]
    return []


def single_agent_reply(client, ctx, human_turns, attempts=4):
    """One direct call -> a burst of 1-N messages. Ignores `human_turns`
    (present only so every responder shares the same signature, see AGENTS):
    ctx['history'] already carries everything this architecture is allowed to
    see, and unlike multi_agent_reply it keeps no structured state (positions/
    settled_issues) across the call.
    """
    system, user = build_single_prompt(ctx)

    traces: list[dict] = []
    for attempt in range(1, attempts + 1):
        raw = None
        try:
            raw, meta = ta.call_json(client, MODEL_ID, system, user, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
            messages = raw.get("messages")
            errors = validate_burst(messages)
            traces.append({"attempt": attempt, "raw": raw, "validation_errors": errors, **meta})
            if not errors:
                return [str(m).strip() for m in messages], {"realizer_attempts": traces}
            user += f"\nPrevious output errors: {'; '.join(errors)}. Return a fresh valid JSON messages array."
        except (ValueError, TypeError, AttributeError, json.JSONDecodeError) as exc:
            traces.append({"attempt": attempt, "raw": raw, "validation_errors": [str(exc)]})

    fallback = ["ok"]
    traces.append({"attempt": "deterministic_fallback", "raw": {"messages": fallback}, "validation_errors": []})
    return fallback, {"realizer_attempts": traces}


# ---------------------------------------------------------------------------
# 2a''. AGENT — single_rag: single_agent.py's turn-grouped, calculator-backed
# single agent, with RAG retrieval (state-matched real human turns as
# exemplars -- rag_align.build_bank/retrieve/format_exemplars). Ported from
# Urmi_assmebledall's step5_replay.turn_reply, this branch's own
# architecture-naming convention instead of her run_name()/condition system.
# ---------------------------------------------------------------------------

RAG_K = 5             # exemplars retrieved per turn

_rag_bank = None
_rag_folds = None


def _rag_setup(human_turns):
    """Build the exemplar bank + leakage-safe retrieval pools once per run (not
    once per context -- every context in this run shares the same bank)."""
    global _rag_bank, _rag_folds
    if _rag_bank is None:
        _rag_bank = ra.build_bank(human_turns, actor_col="Role")
        _rag_folds = ra.two_fold_split(human_turns)
    return _rag_bank, _rag_folds


def pool_for(ctx, folds):
    """Which human dyads may this context retrieve from?

    Two guards, both necessary:
      1. Opposite half   - a context from H1 retrieves only from H2, so no model
                           reply is informed by the human half it is compared to.
      2. Never its OWN dyad - without this, retrieval can surface the exact turn
                           the human wrote at this very context, which would
                           inflate turn-level agreement into meaninglessness.

    Guard 2 matters even though guard 1 exists: a dyad can sit in the pool it
    draws from whenever the halves are unbalanced or a game is not in H1.
    """
    pool = folds["M2_pool"] if ctx["game_id"] in set(folds["H1"]) else folds["M1_pool"]
    return [g for g in pool if g != ctx["game_id"]]


def _prior_messages(human_turns: pd.DataFrame, game_id, turn_idx) -> list[dict]:
    """Every real message before this turn, as single_agent.state_from_turns expects."""
    g = human_turns[(human_turns.GameId == game_id) & (human_turns.turn_idx < turn_idx)]
    g = g.sort_values(["turn_idx", "message_index"])
    return [{"role": str(r.Role), "text": str(r.Text), "turn_idx": int(r.turn_idx)}
            for r in g.itertuples()]


def single_rag_reply(client, ctx, human_turns, attempts=4):
    """One single-agent burst for one human turn context, WITH RAG retrieval.

    Accept is allowed whenever the real transcript already holds a complete package
    from the other side that beats this speaker's outside option; walk_away only in
    the final round. Nothing is forced: replay answers one turn against real prior
    context and does not own the schedule.
    """
    bank, folds = _rag_setup(human_turns)
    pool = pool_for(ctx, folds)       # leakage guard: a context from H1 retrieves only from H2
    q = {"round": ctx["round"], "phase": ctx["phase"], "role": ctx["role"],
         "prev_own": ctx["prev_own"], "history": ctx["history"]}
    rng = np.random.default_rng(abs(hash((ctx["game_id"], ctx["turn_idx"]))) % 2**31)
    ex = ra.retrieve(bank, q, pool, k=RAG_K, mode="matched", rng=rng)

    shown = ctx["is_recipient"] and ctx["round"] >= BATNA_ROUND
    batna = BATNA_NEW if shown else BATNA_INITIAL
    state = sa.state_from_turns(_prior_messages(human_turns, ctx["game_id"], ctx["turn_idx"]))
    where = (f"Round {ctx['round']} of {N_ROUNDS}, "
             f"turn {ctx['msg_in_round']} of this round "
             f"(turn {ctx['msg_pos']} of the negotiation so far)")
    messages, action, info = sa.reply(
        client, MODEL_ID, ctx["role"], round_index=ctx["round"], n_rounds=N_ROUNDS, batna=batna,
        state=state, where=where, history=ctx["history"],
        exemplar_block=ra.format_exemplars(ex), batna_update=shown,
        allow_walk_away=ctx["round"] >= N_ROUNDS, opening=ctx["round"] == 0,
        temperature=TEMPERATURE, attempts=attempts)
    return messages, {"realizer_attempts": info["attempts"], "action": action,
                      "calculator": info["calculator"],
                      "exemplars": ex["text"].tolist() if len(ex) else [],
                      "retrieved_behaviors": ex["_behavior"].tolist() if len(ex) else []}


# ---------------------------------------------------------------------------
# 2a'. AGENT — single_personality: single_agent_reply's system prompt plus a
# Big-Five personality trait profile (Kwon et al., EMNLP 2025 -- see
# personality.py), sampled once per (game_id, role) so it stays IDENTICAL
# across every independent turn-context of the same dyad/role, the same way
# a real person's personality doesn't change mid-negotiation. Seeded off
# (game_id, role) rather than any shared RNG, so it needs no state carried
# between contexts -- the same guarantee context_state_and_turns gives
# multi_agent_reply's state, applied to a value that's per-dyad, not per-turn.
# ---------------------------------------------------------------------------

PERSONALITY_ADJECTIVES = pz.load_personality_adjectives()
N_PERSONALITY_ADJECTIVES = 3
PERSONALITY_INTENSITY = None   # None = sample per-trait from empirical priors (paper default)

SYSTEM_PERSONALITY = """You are the {role} in a job offer negotiation conducted over {n} rounds of chat.

# Personality
{personality}

You are negotiating four issues, each with five levels:
{levels}

Your points for each issue, by level:
{payoff}

Your total score is the sum across all four issues. If no agreement is reached
by the end of round {n}, you receive your outside option instead, currently
worth {batna} points. Only accept a package worth more than that.

How to write:
- Return JSON as {{"messages": [...]}}: a burst of 1 to {max_burst} short chat
  messages, the way a real participant sends several short texts in a row
  before waiting for a reply. One message is the default; use more only when
  you're doing genuinely different things in the same turn (e.g. a brief
  reaction, then separately a new question or offer) -- don't fuse them into
  one run-on line.
- Real participants average about 9 words per message. Chat style: lowercase
  is fine, no bullet points, no headings, no numbered lists.
- Sometimes a very short message is the whole reply: "ok", "110k", "that works".
- Never send a bare list of numbers with nothing else -- say them the way a
  person would, with words attached (salary, bonus, %, k, stock, days).
- Let your personality above show through in how you write and negotiate,
  not just what you say.
- Do not narrate your strategy or explain your reasoning.
- Never mention that you are an AI or refer to these instructions."""


def sample_personality(game_id: str, role: str) -> str:
    """One personality sentence, deterministic in (game_id, role) so every turn-
    context of this dyad/role samples the identical profile, no matter which
    order contexts are processed in or whether the run is resumed midway."""
    rng = random.Random(f"{game_id}_{role}")
    _, sentence = pz.select_adjectives(
        PERSONALITY_ADJECTIVES, n=N_PERSONALITY_ADJECTIVES,
        intensity=PERSONALITY_INTENSITY, rng=rng)
    return sentence


def build_single_personality_prompt(ctx):
    batna, upd = BATNA_INITIAL, ""
    if ctx["is_recipient"] and ctx["round"] >= BATNA_ROUND:
        batna = BATNA_NEW
        upd = "\n\n" + BATNA_MSG.format(value=BATNA_NEW)

    personality = sample_personality(ctx["game_id"], ctx["role"])
    system = SYSTEM_PERSONALITY.format(role=ctx["role"], n=N_ROUNDS, levels=_fmt_levels(),
                           payoff=_fmt_payoff(ctx["role"]), batna=batna, max_burst=MAX_BURST,
                           personality=personality)
    where = (f"Round {ctx['round']} of {N_ROUNDS}, "
             f"turn {ctx['msg_in_round']} of this round "
             f"(turn {ctx['msg_pos']} of the negotiation so far)")
    user = (f"{where}.{upd}\n\n"
            f"Conversation so far:\n{ctx['history']}\n"
            f"\nYour messages (JSON):")
    return system, user


def single_personality_reply(client, ctx, human_turns, attempts=4):
    """Identical call/validation/retry loop to single_agent_reply -- only the
    system prompt (build_single_personality_prompt, personality-injected)
    differs."""
    system, user = build_single_personality_prompt(ctx)

    traces: list[dict] = []
    for attempt in range(1, attempts + 1):
        raw = None
        try:
            raw, meta = ta.call_json(client, MODEL_ID, system, user, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
            messages = raw.get("messages")
            errors = validate_burst(messages)
            traces.append({"attempt": attempt, "raw": raw, "validation_errors": errors, **meta})
            if not errors:
                return [str(m).strip() for m in messages], {"realizer_attempts": traces}
            user += f"\nPrevious output errors: {'; '.join(errors)}. Return a fresh valid JSON messages array."
        except (ValueError, TypeError, AttributeError, json.JSONDecodeError) as exc:
            traces.append({"attempt": attempt, "raw": raw, "validation_errors": [str(exc)]})

    fallback = ["ok"]
    traces.append({"attempt": "deterministic_fallback", "raw": {"messages": fallback}, "validation_errors": []})
    return fallback, {"realizer_attempts": traces}


# ---------------------------------------------------------------------------
# 2b. AGENT — multi, two_agents.py's StrategyPlannerAgent + LanguageRealizerAgent.
#
# Every context goes through the planner first: it decides the action
# (including `greet`, when nothing has been said yet -- see
# two_agents.plan_action) and the realizer writes its own natural burst of
# 1-5 chat messages for that one decision, mirroring how real dyads send
# several short messages in a row before the other side replies. There is no
# separate reaction/content split (that was `react_first`, retired once
# turn-grouping made it unnecessary: a real burst can already lead with a
# bare acknowledgment and follow with the substantive content, without being
# forced to).
# ---------------------------------------------------------------------------

def context_state_and_turns(human_turns: pd.DataFrame, game_id: str, turn_idx: int):
    """Rebuild exactly what a live two_agents.py run would have accumulated by this
    point, from the REAL human transcript alone -- not from any model call this
    script has made, so nothing carries across contexts. `human_turns` is
    message-level (group_into_turns output); `turn_idx` is the TURN position
    being replied to -- everything with turn_idx strictly before it is "prior".
    """
    g = human_turns[human_turns.GameId == game_id]
    prior = g[g.turn_idx < turn_idx]
    turns = [{"role": str(r.Role), "text": str(r.Text)} for r in prior.itertuples()]

    state = ta.initial_shared_state(ROLES)
    for role in ROLES:
        role_msgs = prior[prior.Role == role]["Text"].astype(str).tolist()
        state["current_position_by_role"][role] = ta.parse_messages(role_msgs)
    state = ta._recompute_settled(state)
    return turns, state


def multi_agent_reply(client, ctx, human_turns):
    """Produce one model reply (a burst of 1-N messages) for this context.

    Every context goes through the planner first -- there is no round-based or
    content-based pre-filter deciding "this is still greeting." The planner reads
    the actual last line (recent_conversation) and chooses `greet` itself when
    appropriate, with a hard backstop in validate_planner_decision that blocks
    `greet` once real numbers are already on the table. This is deliberately more
    than a numeric-content check: "what do you value most?" has no numbers but is
    real content and should route to ask/offer/hold, not greet -- only the model
    reading the line can tell those apart.
    """
    role, opponent = ctx["role"], next(r for r in ROLES if r != ctx["role"])
    turns, state = context_state_and_turns(human_turns, ctx["game_id"], ctx["turn_idx"])
    batna = BATNA_NEW if (ctx["is_recipient"] and ctx["round"] >= BATNA_ROUND) else BATNA_INITIAL

    own_position = state["current_position_by_role"][role]
    opponent_position = state["current_position_by_role"][opponent]
    shortlist = ta.rank_offers(role, batna, own_position, opponent_position, state["settled_issues"], limit=12)

    decision, planner_traces = ta.plan_action(
        client, MODEL_ID, role, opponent, ctx["round"], batna, turns, state, shortlist,
        must_resolve=False, memory=[], final_round=N_ROUNDS,
    )

    if decision["action"] == "greet":
        messages, realizer_traces = ta.realize_greeting(client, MODEL_ID, role, turns, examples=[])
    else:
        messages, realizer_traces = ta.realize_action(client, MODEL_ID, role, decision, turns, examples=[], attempts=4)

    trace = {
        "batna": batna, "settled_issues": state["settled_issues"], "shortlist_size": len(shortlist),
        "planner_decision": decision, "planner_attempts": planner_traces, "realizer_attempts": realizer_traces,
    }
    return messages, trace


# ---------------------------------------------------------------------------
# 3. Run — architecture-agnostic. Add a new agent by writing one more
#    responder(client, ctx, human_turns) -> (messages: list[str], trace_dict_or_None)
#    and registering it here.
# ---------------------------------------------------------------------------

AGENTS = {
    "single": single_agent_reply,
    "multi": multi_agent_reply,
    "single_personality": single_personality_reply,
    "single_rag": single_rag_reply,
}


def run_dir(agent_name: str) -> str:
    output_key = agent_name if MODEL == lb.DEFAULT_MODEL else f"{agent_name}_{MODEL}"
    return f"{RUNS_DIR}/{output_key}"


def run(agent_name, contexts, human_turns, limit=None):
    responder = AGENTS[agent_name]
    rdir = run_dir(agent_name)
    os.makedirs(rdir, exist_ok=True)
    path = f"{rdir}/replay.jsonl"
    trace_path = f"{rdir}/private_traces.jsonl"

    done = set()
    if os.path.exists(path):
        done = {(json.loads(l)["game_id"], json.loads(l)["turn_idx"])
                for l in open(path)}
        print(f"resuming: {len(done)} contexts already done")

    client = get_client()
    todo = contexts[:limit] if limit else contexts

    with open(path, "a") as f, open(trace_path, "a") as tf:
        for n, ctx in enumerate(todo):
            if (ctx["game_id"], ctx["turn_idx"]) in done:
                continue
            try:
                messages, trace = responder(client, ctx, human_turns)
            except Exception as exc:
                print(f"  FAILED {ctx['game_id']}#{ctx['turn_idx']}: {exc!r}", flush=True)
                continue

            f.write(json.dumps({
                **{k: ctx[k] for k in ("game_id", "turn_idx", "round", "phase", "role", "human_text",
                                       "human_behavior", "human_messages", "human_behaviors", "prev_own",
                                       "prev_speaker", "batna_target", "is_recipient", "msg_pos", "msg_in_round",
                                       "burst_size")},   # the HUMAN turn's burst size, from build_turn_contexts
                "model_messages": messages,
                "strategic_turn_id": f"{ctx['game_id']}_turn_{ctx['turn_idx']:02d}",
                "model_burst_size": len(messages),
                "batna_shown": ctx["is_recipient"] and ctx["round"] >= BATNA_ROUND,
            }) + "\n")
            f.flush()
            if trace is not None:
                tf.write(json.dumps({"game_id": ctx["game_id"], "turn_idx": ctx["turn_idx"], **trace}) + "\n")
                tf.flush()
            if n % 25 == 0:
                print(f"  {n}/{len(todo)}", flush=True)

    export(agent_name)


def export(agent_name):
    """Write two files into replay/<agent_name>/:
      replay_pairs.csv       one row per context (per TURN): human turn beside model turn.
      replay_unlabeled.csv   model MESSAGES only, in ds.manual schema, so the classifier
                              and rag_align run on them unchanged -- one row per message,
                              turn_idx/message_index/burst_size carried along, since
                              behavior_mix/mix_JSD are inherently message-level even
                              though pairing for turn-level agreement (step6_paired.py)
                              needs one label per turn.

    NOTE the GameId trick: model turns keep the GameId of the context they came
    from. That gives you 50 'model dyads' matching your 50 human dyads, so every
    metric in rag_align works with no code changes, and both sides of the
    comparison are the same size.
    """
    rdir = run_dir(agent_name)
    rows = [json.loads(l) for l in open(f"{rdir}/replay.jsonl")]
    pairs = pd.DataFrame(rows)
    pairs.to_csv(f"{rdir}/replay_pairs.csv", index=False)

    exploded = pairs.explode("model_messages", ignore_index=True)
    exploded["message_index"] = exploded.groupby(["game_id", "turn_idx"]).cumcount() + 1
    exploded["burst_size"] = exploded.groupby(["game_id", "turn_idx"])["model_messages"].transform("size")
    man = pd.DataFrame({
        "GameId": exploded["game_id"], "ProlificId": exploded["role"], "Role": exploded["role"],
        "Round": exploded["round"], "Phase": exploded["phase"], "Text": exploded["model_messages"],
        "Behavior": np.nan, "turn_idx": exploded["turn_idx"],
        "message_index": exploded["message_index"], "burst_size": exploded["burst_size"],
    }).sort_values(["GameId", "turn_idx", "message_index"]).reset_index(drop=True)
    man.to_csv(f"{rdir}/replay_unlabeled.csv", index=False)

    hw = pairs.human_messages.apply(lambda ms: sum(len(str(m).split()) for m in ms))
    mw = pairs.model_messages.apply(lambda ms: sum(len(str(m).split()) for m in ms))
    print(f"\n{len(pairs)} paired turns ({man.shape[0]} model messages), "
          f"{pairs.game_id.nunique()} dyads")
    print(f"  burst size   human mean {pairs.burst_size.mean():.2f}   "
          f"model mean {pairs.model_messages.apply(len).mean():.2f}")
    print(f"  words   human {hw.mean():.1f} (sd {hw.std():.1f}, "
          f"<4w {(hw<4).mean():.0%})")
    print(f"  words   model {mw.mean():.1f} (sd {mw.std():.1f}, "
          f"<4w {(mw<4).mean():.0%})")
    print(f"  pre contexts {(pairs.phase==ra.PRE).sum()}, "
          f"post {(pairs.phase==ra.POST).sum()}")
    if "is_recipient" in pairs.columns:
        for lab, sub in [("recipient", pairs[pairs.is_recipient]),
                         ("counterpart", pairs[~pairs.is_recipient])]:
            if len(sub) == 0:
                continue
            pre = sub[sub.phase == ra.PRE]
            post = sub[sub.phase == ra.POST]
            print(f"  {lab:12s} pre {len(pre):4d}  post {len(post):4d}")
    print(f"\n  -> classify/classify_replay.py --agent {agent_name} --submit")


def peek(agent_name, contexts, human_turns):
    for ctx in [contexts[2], contexts[len(contexts) // 2], contexts[-3]]:
        print("=" * 70)
        print(f"{ctx['game_id']}  round {ctx['round']}  {ctx['phase']}  "
              f"role {ctx['role']}  burst_size={ctx['burst_size']}  "
              f"{'RECIPIENT' if ctx['is_recipient'] else 'counterpart'}")
        if agent_name == "multi":
            turns, state = context_state_and_turns(human_turns, ctx["game_id"], ctx["turn_idx"])
            print(f"settled_issues so far: {state['settled_issues']}")
            print(f"own position: {state['current_position_by_role'][ctx['role']]}")
            print("-" * 70)
            for t in turns[-6:]:
                print(f"  {t['role']}: {t['text']}")
        else:
            _, u = build_single_prompt(ctx)
            print("-" * 70)
            print(u[:1400])
        print("-" * 70)
        print(f"HUMAN ACTUALLY WROTE ({list(zip(ctx['human_behaviors'], ctx['human_messages']))}):\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="single", choices=list(AGENTS),
                    help="single: one direct system-prompted call. "
                         "multi: two_agents.py's planner+realizer.")
    ap.add_argument("--peek", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dyads", type=int, default=None,
                    help="cap --run to the first N human dyads (by GameId order); "
                         "omit for the full run. ignored by --smoke/--peek.")
    ap.add_argument("--human-csv", default=None,
                    help="alternate human transcript CSV (default: HUMAN_CSV, "
                         "i.e. data/human_data.csv).")
    ap.add_argument("--model", default="gpt-5.4", choices=list(lb.MODELS),
                    help="generating LLM backend -- see llm_backends.py. Output lands in "
                         "replay/<agent>/ for gpt-5.4 (unchanged), replay/<agent>_<model>/ otherwise.")
    ap.add_argument("--out-dir", default=RUNS_DIR,
                    help=f"base directory for output (default: {RUNS_DIR}, the shipped results). "
                         "Point this elsewhere (e.g. run_pipeline.py --full's default) to regenerate "
                         "without touching the committed data.")
    a = ap.parse_args()
    MODEL = a.model
    MODEL_ID = lb.model_id(MODEL)
    RUNS_DIR = a.out_dir

    human = load_human(a.human_csv)
    human_turns = group_into_turns(human)
    batna_targets = load_batna_targets()
    contexts = build_turn_contexts(human_turns, batna_targets)

    if a.dyads:
        keep = list(dict.fromkeys(c["game_id"] for c in contexts))[:a.dyads]
        keep_set = set(keep)
        contexts = [c for c in contexts if c["game_id"] in keep_set]
        print(f"--dyads {a.dyads}: restricted to {len(keep)} dyads {keep}")

    print(f"human   : {len(human)} messages, {human.GameId.nunique()} dyads")
    print(f"contexts: {len(contexts)} turns, agent={a.agent} "
          f"(burst sizes: {pd.Series([c['burst_size'] for c in contexts]).value_counts().sort_index().to_dict()})")
    rec = sum(c["is_recipient"] for c in contexts)
    print(f"  recipient contexts {rec}   counterpart {len(contexts)-rec}")
    print("  (the counterpart split is the harder test: they were never told)\n")

    if a.peek:
        peek(a.agent, contexts, human_turns)
    elif a.smoke:
        run(a.agent, contexts, human_turns, limit=20)
    elif a.run:
        run(a.agent, contexts, human_turns)
    else:
        print(__doc__)
