"""
step2_generate.py — free-running dyad generation, any agent architecture
===========================================================================

A full 6-round negotiation generated end to end -- both sides invented from
scratch, no real human transcript mid-dialogue (unlike step5_replay.py,
which replays one turn at a time against real human context). No RAG, no
human examples in the prompt, in either architecture.

--agent single   this file's own design: one direct system-prompted call
                 per message (MSGS_PER_ROUND_PER_PARTY per party per round).
--agent multi    two_agents.py's StrategyPlannerAgent + LanguageRealizerAgent,
                 via simulate_multi_agent_dyad -- persistent state carried
                 for the whole dialogue (this is the one place in the repo
                 where that statefulness actually gets to operate; replay
                 resets it every context by design -- see step5_replay.py).

OUTPUT LAYOUT -- one folder per agent, mirroring replay/<agent>/ from the
replay pipeline:
    freerun/single/run.jsonl, private_traces.jsonl (empty -- single has none),
            run_unlabeled.csv
    freerun/multi/  same three, private_traces.jsonl populated (planner
            decisions, realizer attempts, state before/after -- one line
            per dyad, {"game_id", "traces": [...]})

USAGE
    export OPENAI_API_KEY=sk-...
    python generation/step2_generate.py --agent single --smoke   # 3 dyads, checks length
    python generation/step2_generate.py --agent single --run     # full N_DYADS, resumes
    python generation/step2_generate.py --agent multi  --run
"""

import argparse
import json
import os
import random
import sys
import time

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
# CONFIG — EDIT THIS BLOCK
# ===========================================================================

HUMAN_CSV = "data/human_data.csv"   # what the model is evaluated against

# --- Payoffs, from Table A2 of the CHI submission -------------------------
# Utilities are additive across issues; level l in 1..5 -> index 0..4.
# Verified: HR totals span 11-63, candidate 14-70, matching Appendix A.2.
# Note the stock function for HR is 4-(6-l) = l-2, which is NEGATIVE at
# level 1. That is what the paper specifies and it reproduces the stated
# range exactly, so it is intentional, not a typo.
ISSUES = ["salary", "bonus", "stock options", "vacation days"]

_L = [1, 2, 3, 4, 5]
PAYOFF = {
    "HR": {                                      # 3(6-l), 3(6-l), 4-(6-l), 6(6-l)
        "salary":        [3 * (6 - l) for l in _L],   # 15 12  9  6  3
        "bonus":         [3 * (6 - l) for l in _L],   # 15 12  9  6  3
        "stock options": [4 - (6 - l) for l in _L],   # -1  0  1  2  3
        "vacation days": [6 * (6 - l) for l in _L],   # 30 24 18 12  6
    },
    "Employee": {                                # 6l, 3l, 2l, 3l
        "salary":        [6 * l for l in _L],         #  6 12 18 24 30
        "bonus":         [3 * l for l in _L],         #  3  6  9 12 15
        "stock options": [2 * l for l in _L],         #  2  4  6  8 10
        "vacation days": [3 * l for l in _L],         #  3  6  9 12 15
    },
}

# Real-world values shown to participants (Table A3). Included in the prompt
# so agents negotiate over dollars and days, not abstract levels.
LEVELS = {
    "salary":        ["$80,000", "$90,000", "$100,000", "$110,000", "$120,000"],
    "bonus":         ["0%", "5%", "10%", "15%", "20%"],
    "stock options": ["$0", "$5,000", "$10,000", "$15,000", "$20,000"],
    "vacation days": ["10", "15", "20", "25", "30"],
}

# --- Roles ----------------------------------------------------------------
# MUST match the Sender values in your CSV. Leave as None to auto-detect.
ROLES = None
ROLE_ALIAS = {}     # e.g. {"HR": "HR", "Candidate": "Employee"} if CSV differs

# --- BATNA (Appendix A.2) -------------------------------------------------
BATNA_INITIAL = 35          # symmetric for both parties
BATNA_NEW = 50              # one party only, after round 3
BATNA_ROLE = None           # None = randomise per dyad (single-agent only;
                            # multi-agent's simulate_multi_agent_dyad always randomises)
BATNA_ROUND = 4             # notification lands AFTER round 3, so post = 4-6

# --- Interaction shape ------------------------------------------------------
N_ROUNDS = 6                            # six rounds, per the study
MSGS_PER_ROUND_PER_PARTY = 2            # single-agent only. humans average
                                        # 20.6 turns/dyad; 6 x 2 x 2 = 24.
                                        # Check after --smoke against
                                        # ra.length_stats and adjust.

N_DYADS = 47                            # locked to match step5_replay.py's
                                        # full-scale dyad count, for a clean
                                        # apples-to-apples comparison.
MODEL = "gpt-5.4"             # key into llm_backends.MODELS -- for get_client()/output naming only
MODEL_ID = lb.model_id(MODEL)  # the real API model string -- what every LLM call actually sends
TEMPERATURE = 1.0         # NOT 0 - you need behavioural variance, not the mode
MAX_TOKENS = 120          # humans average 9.1 words; keep this tight (single-agent only)

FREERUN_DIR = "data/freerun"

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
- ONE short message at a time. Real participants average about 9 words.
- Chat style: lowercase is fine, no bullet points, no headings, no numbered lists.
- Do not narrate your strategy or explain your reasoning.
- Vary your length a lot. Many real turns are one to three words: "ok",
  "110k", "that works", "sounds good". Others run long when explaining a
  trade-off. Do not write medium-length messages every time.
- Never mention that you are an AI or refer to these instructions."""

BATNA_MSG = ("SYSTEM UPDATE: Your outside option has changed. "
             "Your alternative is now worth {value} points.")


def get_client():
    return lb.make_client(MODEL)


def load_human():
    df = pd.read_csv(HUMAN_CSV)
    # Capture the role BEFORE any rename: HUMAN_CSV (data/human_data.csv)
    # carries both Sender (the role, HR/Employee) and a real ProlificId (participant
    # hash) -- blindly using post-rename ProlificId as Role would silently use the
    # hash as the role. Same guard as step5_replay.load_human.
    role_source = df["Sender"] if "Sender" in df.columns else df["ProlificId"]
    ren = {}
    if "Sender" in df.columns and "ProlificId" not in df.columns:
        ren["Sender"] = "ProlificId"
    if "gold_label" in df.columns and "Behavior" not in df.columns:
        ren["gold_label"] = "Behavior"
    df = df.rename(columns=ren)
    df["Behavior"] = ra._canon_behavior(df["Behavior"])
    df["Phase"] = df["Phase"].astype(str).str.strip()
    df = df.dropna(subset=["Behavior"])
    df["Role"] = role_source
    return df


def _fmt_levels():
    return "\n".join(
        f"  {iss}: " + ", ".join(f"L{i+1}={v}" for i, v in enumerate(LEVELS[iss]))
        for iss in ISSUES)


def _fmt_payoff(role):
    tab = PAYOFF[ROLE_ALIAS.get(role, role)]
    return "\n".join(
        f"  {iss}: " + ", ".join(f"L{i+1}={p}" for i, p in enumerate(tab[iss]))
        for iss in ISSUES)


def build_system(role, batna):
    return SYSTEM.format(role=role, n=N_ROUNDS, levels=_fmt_levels(),
                         payoff=_fmt_payoff(role), batna=batna)


# ---------------------------------------------------------------------------
# AGENT — single: this file's own design, one direct call per message.
# ---------------------------------------------------------------------------

def simulate_dyad_single(client, gid, seed):
    """One full free-running dyad. Returns (dialogue, trace_or_None) -- single
    has no private trace, unlike multi's planner/realizer attempts."""
    rng = np.random.default_rng(seed)
    hist, turns = [], []
    target = BATNA_ROLE or ROLES[int(rng.integers(2))]   # who gets the change
    batna = {r: BATNA_INITIAL for r in ROLES}

    for rnd in range(1, N_ROUNDS + 1):
      for _msg in range(MSGS_PER_ROUND_PER_PARTY):
        for role in ROLES:
            phase = ra.PRE if rnd < BATNA_ROUND else ra.POST
            convo = "\n".join(f"{t['role']}: {t['text']}" for t in hist[-8:])

            upd = ""
            if role == target and rnd == BATNA_ROUND and _msg == 0:
                batna[role] = BATNA_NEW
                upd = "\n\n" + BATNA_MSG.format(value=BATNA_NEW)

            user = (f"Round {rnd} of {N_ROUNDS}.{upd}\n\n"
                    f"Conversation so far:\n{convo or '(you speak first)'}\n"
                    f"\nYour message:")

            for attempt in range(4):
                try:
                    r = client.chat.completions.create(
                        model=MODEL_ID, temperature=TEMPERATURE,
                        # client comes from llm_backends.make_client() -- its shims take
                        # `max_tokens` and translate per-provider internally.
                        max_tokens=MAX_TOKENS,
                        messages=[{"role": "system",
                                   "content": build_system(role, batna[role])},
                                  {"role": "user", "content": user}])
                    txt = r.choices[0].message.content.strip()
                    break
                except Exception as e:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)

            rec = {"actor_id": f"{gid}_{role}", "role": role,
                   "round": rnd, "phase": phase, "text": txt}
            turns.append(rec); hist.append(rec)

    return {"game_id": gid, "turns": turns, "batna_target": target}, None


# ---------------------------------------------------------------------------
# AGENT — single_personality: simulate_dyad_single plus a Big-Five personality
# trait profile (Kwon et al., EMNLP 2025 -- see personality.py), sampled once
# per role per dyad, since a person's personality doesn't change mid-
# negotiation. Seeded off (gid, role) -- the same scheme step5_replay.py's
# single_personality_reply uses -- so the same dyad/role samples an identical
# profile in either pipeline, though nothing here depends on that coincidence.
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
- ONE short message at a time. Real participants average about 9 words.
- Chat style: lowercase is fine, no bullet points, no headings, no numbered lists.
- Do not narrate your strategy or explain your reasoning.
- Vary your length a lot. Many real turns are one to three words: "ok",
  "110k", "that works", "sounds good". Others run long when explaining a
  trade-off. Do not write medium-length messages every time.
- Let your personality above show through in how you write and negotiate,
  not just what you say.
- Never mention that you are an AI or refer to these instructions."""


def sample_personality(game_id: str, role: str) -> str:
    """One personality sentence, deterministic in (game_id, role)."""
    rng = random.Random(f"{game_id}_{role}")
    _, sentence = pz.select_adjectives(
        PERSONALITY_ADJECTIVES, n=N_PERSONALITY_ADJECTIVES,
        intensity=PERSONALITY_INTENSITY, rng=rng)
    return sentence


def build_system_personality(role, batna, personality):
    return SYSTEM_PERSONALITY.format(role=role, n=N_ROUNDS, levels=_fmt_levels(),
                                     payoff=_fmt_payoff(role), batna=batna,
                                     personality=personality)


def simulate_dyad_single_personality(client, gid, seed):
    """simulate_dyad_single, but every call's system prompt carries a personality
    profile sampled once per role for this dyad."""
    rng = np.random.default_rng(seed)
    hist, turns = [], []
    target = BATNA_ROLE or ROLES[int(rng.integers(2))]
    batna = {r: BATNA_INITIAL for r in ROLES}
    personalities = {r: sample_personality(gid, r) for r in ROLES}

    for rnd in range(1, N_ROUNDS + 1):
      for _msg in range(MSGS_PER_ROUND_PER_PARTY):
        for role in ROLES:
            phase = ra.PRE if rnd < BATNA_ROUND else ra.POST
            convo = "\n".join(f"{t['role']}: {t['text']}" for t in hist[-8:])

            upd = ""
            if role == target and rnd == BATNA_ROUND and _msg == 0:
                batna[role] = BATNA_NEW
                upd = "\n\n" + BATNA_MSG.format(value=BATNA_NEW)

            user = (f"Round {rnd} of {N_ROUNDS}.{upd}\n\n"
                    f"Conversation so far:\n{convo or '(you speak first)'}\n"
                    f"\nYour message:")

            for attempt in range(4):
                try:
                    r = client.chat.completions.create(
                        model=MODEL_ID, temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        messages=[{"role": "system",
                                   "content": build_system_personality(role, batna[role], personalities[role])},
                                  {"role": "user", "content": user}])
                    txt = r.choices[0].message.content.strip()
                    break
                except Exception as e:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)

            rec = {"actor_id": f"{gid}_{role}", "role": role,
                   "round": rnd, "phase": phase, "text": txt}
            turns.append(rec); hist.append(rec)

    return {"game_id": gid, "turns": turns, "batna_target": target,
            "personality": personalities}, None


# ---------------------------------------------------------------------------
# AGENT — multi: two_agents.py's planner + realizer, full persistent state
# for the whole dialogue. This is the one place in the repo where that
# statefulness actually gets exercised -- the replay pipeline resets it
# fresh from the real transcript every single context, by design (see
# step5_replay.py's module docstring for why that's a real limitation of
# replay specifically, not of this architecture).
# ---------------------------------------------------------------------------

def no_retrieve_examples(role, round_index, phase, behavior, history):
    """ta.simulate_multi_agent_dyad still takes a retrieve_examples callable
    (see its docstring) -- this is the no-op version: no exemplars, ever."""
    return [], []


def simulate_dyad_multi(client, gid, seed):
    """Returns (dialogue, traces) -- traces is two_agents.py's full per-turn
    planner/realizer record, written to private_traces.jsonl."""
    rng = random.Random(seed)
    batna_target = ROLES[rng.randrange(2)]
    config = ta.MultiAgentConfig(
        model=MODEL_ID, initial_batna=BATNA_INITIAL, new_batna=BATNA_NEW,
        batna_round=BATNA_ROUND, final_round=N_ROUNDS,
    )
    public, traces = ta.simulate_multi_agent_dyad(
        client=client, game_id=gid, roles=tuple(ROLES), rng=rng,
        batna_target=batna_target, retrieve_examples=no_retrieve_examples, config=config,
    )
    return public, traces


# ---------------------------------------------------------------------------
# AGENT — single_rag: single_agent.py's turn-grouped, calculator-backed single
# agent, with RAG retrieval (state-matched real human turns as exemplars --
# rag_align.build_bank/retrieve/format_exemplars). Ported from
# Urmi_assmebledall's step2_generate.simulate_turn_dyad, this branch's own
# architecture-naming convention instead of her run_name()/condition system.
# ---------------------------------------------------------------------------

RAG_K = 5                   # exemplars retrieved per turn
RAG_HISTORY_TURNS = 8        # same window step5_replay uses

_rag_bank = None
_rag_folds = None


def _rag_setup(human):
    """Build the exemplar bank + leakage-safe retrieval pools once per run (not
    once per dyad -- every dyad in this run shares the same bank)."""
    global _rag_bank, _rag_folds
    if _rag_bank is None:
        _rag_bank = ra.build_bank(human, actor_col="Role")
        _rag_folds = ra.two_fold_split(human)
    return _rag_bank, _rag_folds


def simulate_dyad_single_rag(client, gid, seed):
    """One free-running dyad of the turn-grouped single agent WITH RAG retrieval.
    Returns (dialogue, traces) -- traces is single_agent.reply()'s per-turn
    calculator facts + every LLM attempt, written to private_traces.jsonl."""
    rng = np.random.default_rng(seed)
    dyad_index = seed - SEED_BASE["single_rag"]
    bank, folds = _rag_bank, _rag_folds
    pool = list(folds["M1_pool"] if dyad_index % 2 == 0 else folds["M2_pool"])

    target = BATNA_ROLE or ROLES[int(rng.integers(2))]
    order = list(ROLES)
    schedule = [(0, r) for r in order] + [(rnd, r) for rnd in range(1, N_ROUNDS + 1) for r in order]

    turns, traces = [], []
    outcome = {"terminal_action": None, "deal": None}
    for ti, (rnd, role) in enumerate(schedule):
        phase = ra.PRE if rnd < BATNA_ROUND else ra.POST
        batna = BATNA_NEW if (role == target and rnd >= BATNA_ROUND) else BATNA_INITIAL
        history = sa.render_history(turns, RAG_HISTORY_TURNS)
        state = sa.state_from_turns(turns, roles=order)
        in_round = sum(1 for r, _ in schedule[:ti] if r == rnd) + 1

        q = {"round": rnd, "phase": phase, "role": role, "prev_own": None, "history": history}
        ex = ra.retrieve(bank, q, pool, k=RAG_K, mode="matched", rng=rng)

        messages, action, info = sa.reply(
            client, MODEL_ID, role, round_index=rnd, n_rounds=N_ROUNDS, batna=batna, state=state,
            where=(f"Round {rnd} of {N_ROUNDS}, turn {in_round} of this round "
                   f"(turn {ti + 1} of the negotiation so far)"),
            history=history, exemplar_block=ra.format_exemplars(ex),
            batna_update=(role == target and rnd >= BATNA_ROUND),
            must_full_offer=(ti == len(schedule) - 2), must_resolve=(ti == len(schedule) - 1),
            opening=(rnd == 0), enforce_opening=True, temperature=TEMPERATURE)

        strategic_turn_id = f"{gid}_turn_{ti:02d}"
        for mi, m in enumerate(messages, start=1):
            turns.append({"actor_id": f"{gid}_{role}", "role": role, "round": rnd, "phase": phase,
                         "text": m, "turn_idx": ti, "strategic_turn_id": strategic_turn_id,
                         "message_index": mi, "burst_size": len(messages)})
        calc = info["calculator"]
        traces.append({"turn_idx": ti, "retrieved_texts": ex["text"].tolist() if len(ex) else [],
                       "retrieved_behaviors": ex["_behavior"].tolist() if len(ex) else [], **info})

        if action in ("accept", "walk_away"):
            outcome.update(terminal_action=action,
                           deal=calc["pending_package"] if action == "accept" else None)
            break

    public = {
        "game_id": gid, "single_agent_rag_version": "single_rag-ported-from-Urmi_assmebledall",
        "turns": turns, "batna_target": target,
        "message_metrics": ta.burst_metrics(turns),
        "deal": outcome["deal"], "terminal_action": outcome["terminal_action"],
    }
    return public, traces


# ---------------------------------------------------------------------------
# RUN — architecture-agnostic. Add a new agent by writing one more
#   responder(client, gid, seed) -> (dialogue: dict, trace_or_None)
# and registering it here.
# ---------------------------------------------------------------------------

AGENTS = {
    "single": simulate_dyad_single,
    "multi": simulate_dyad_multi,
    "single_personality": simulate_dyad_single_personality,
    "single_rag": simulate_dyad_single_rag,
}
SEED_BASE = {"single": 20260806, "multi": 20260825, "single_personality": 20260901,
            "single_rag": 20261012}


def run_dir(agent_name: str) -> str:
    output_key = agent_name if MODEL == lb.DEFAULT_MODEL else f"{agent_name}_{MODEL}"
    return f"{FREERUN_DIR}/{output_key}"


def run(agent_name, n_dyads, human):
    responder = AGENTS[agent_name]
    if agent_name == "single_rag":
        _rag_setup(human)
    rdir = run_dir(agent_name)
    output_key = agent_name if MODEL == lb.DEFAULT_MODEL else f"{agent_name}_{MODEL}"
    os.makedirs(rdir, exist_ok=True)
    path = f"{rdir}/run.jsonl"
    trace_path = f"{rdir}/private_traces.jsonl"

    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["game_id"] for l in open(path)}
        print(f"resuming: {len(done)} dyads already generated")

    client = get_client()

    with open(path, "a") as f, open(trace_path, "a") as tf:
        for i in range(n_dyads):
            gid = f"{output_key}_{i:03d}"
            if gid in done:
                continue
            seed = SEED_BASE[agent_name] + i   # fresh, deterministic per dyad --
                                               # unlike a single shared RNG advanced
                                               # across the loop, this makes a resumed
                                               # run reproduce identically regardless
                                               # of where it was interrupted.
            try:
                dialogue, trace = responder(client, gid, seed)
                f.write(json.dumps(dialogue) + "\n"); f.flush()
                if trace is not None:
                    tf.write(json.dumps({"game_id": gid, "traces": trace}) + "\n"); tf.flush()
                extra = ""
                if dialogue.get("deal") is not None:
                    extra = f" (deal={'yes' if dialogue['deal'] else 'no'}, terminal={dialogue.get('terminal_action')})"
                print(f"  {gid} ok{extra}", flush=True)
            except Exception as e:
                print(f"  {gid} FAILED {e!r}", flush=True)

    export(agent_name, human)


def export(agent_name, human):
    rdir = run_dir(agent_name)
    dial = [json.loads(l) for l in open(f"{rdir}/run.jsonl")]
    man = ra.to_manual(dial, batna_round=BATNA_ROUND)
    man["Role"] = man["ProlificId"].str.rsplit("_", n=1).str[-1]

    # multi-agent's turns carry extra strategic_turn_id/message_index/burst_size
    # (each planner decision can realize as a 1-5 message burst); single-agent has
    # none of that (one message per call, no bursting) -- bolt the columns on only
    # when present, so downstream classify/report can use turn-level structure for
    # multi without single needing to fake columns it has no real content for.
    burst_rows = [turn for d in dial for turn in d["turns"]]
    if burst_rows and "strategic_turn_id" in burst_rows[0]:
        for col in ("strategic_turn_id", "message_index", "burst_size"):
            man[col] = [turn[col] for turn in burst_rows]
    man.to_csv(f"{rdir}/run_unlabeled.csv", index=False)

    print(f"\n{len(man)} messages, {man.GameId.nunique()} dyads")
    print(f"  human length  : {ra.length_stats(human)}")
    print(f"  {agent_name:6s} length : {ra.length_stats(man)}")
    hw = human.Text.astype(str).str.split().str.len().mean()
    mw = man.Text.astype(str).str.split().str.len().mean()
    if mw > 2.5 * hw:
        print(f"\n  !! {mw:.1f} vs {hw:.1f} words. The model is far more verbose than")
        print("  !! your humans. Length differences drag behaviour codes.")

    if agent_name in ("multi", "single_rag"):
        deals = sum(1 for d in dial if d.get("deal"))
        walk_aways = sum(1 for d in dial if d.get("terminal_action") == "walk_away")
        print(f"  deals reached : {deals}/{len(dial)}")
        print(f"  walk-aways    : {walk_aways}/{len(dial)}")


def main():
    global ROLES, MODEL, MODEL_ID, FREERUN_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="single", choices=list(AGENTS),
                    help="single: one direct system-prompted call per message. "
                         "multi: two_agents.py's planner+realizer, full dialogue state.")
    ap.add_argument("--smoke", action="store_true", help="3 dyads, for checking")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model", default="gpt-5.4", choices=list(lb.MODELS),
                    help="generating LLM backend -- see llm_backends.py. Output lands in "
                         "freerun/<agent>/ for gpt-5.4 (unchanged), freerun/<agent>_<model>/ otherwise.")
    ap.add_argument("--out-dir", default=FREERUN_DIR,
                    help=f"base directory for output (default: {FREERUN_DIR}, the shipped results). "
                         "Point this elsewhere (e.g. run_pipeline.py --full's default) to regenerate "
                         "without touching the committed data.")
    a = ap.parse_args()
    MODEL = a.model
    MODEL_ID = lb.model_id(MODEL)
    FREERUN_DIR = a.out_dir

    human = load_human()
    if ROLES is None:
        ROLES = sorted(human["Role"].dropna().unique().tolist())
        print(f"auto-detected roles from CSV: {ROLES}")
        missing = [r for r in ROLES if ROLE_ALIAS.get(r, r) not in PAYOFF]
        if missing:
            sys.exit(f"No payoff table for {missing}. Set ROLE_ALIAS to map "
                     f"your CSV role names onto {list(PAYOFF)}.")
    print(f"human: {len(human)} msgs, {human.GameId.nunique()} dyads, agent={a.agent}\n")

    if a.smoke:
        run(a.agent, 3, human)
    elif a.run:
        run(a.agent, N_DYADS, human)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
