"""single_agent.py — the turn-grouped single-agent negotiator, RAG-enabled variant.

Ported from Urmi_assmebledall's single_agent.py (which produced the paper's single+RAG
results), adapted onto this repo's own architecture-naming convention (single_rag,
alongside single/multi/single_personality) instead of Urmi's own condition/model
run-naming scheme (run_name/split_run/file_stem, dropped here -- see
step2_generate.py's/step5_replay.py's existing run_dir()/output_key() for the
equivalent on this branch).

Shared by step2_generate.py (free run) and step5_replay.py (replay), so both designs
use byte-identical prompts, payoff calculator and validation. RAG retrieval itself
(rag_align.build_bank/retrieve/format_exemplars) is NOT in this file -- the caller
builds the exemplar_block string and passes it in; this module only knows what to do
with it once built (see WHAT CHANGED's point 2 below).

WHAT CHANGED FROM THE PLAIN SINGLE AGENT (single_agent_reply/simulate_dyad_single)
    1. TURN-GROUPED. One call = one TURN: the model reads everything the other side
       just sent and replies with its own burst of 1-5 messages, the way real
       participants do (and the way two_agents.py already works), rather than one
       message per call.
    2. PAYOFF CALCULATOR. Every turn the prompt carries a private calculator block,
       built from the same arithmetic two_agents.py gives its reasoners
       (score_package, pending_package_for, rank_offers_for_calculator): what the
       other side's complete package is worth to you and whether it beats your
       outside option, what their individual asks are worth, and a short list of
       packages that beat your outside option balanced against what they have asked
       for. The model no longer has to add up the payoff table in its head.
    3. EXPLICIT ACTION. The reply is JSON {"messages": [...], "action": ...} with
       action continue / accept / walk_away, validated against the calculator (you
       cannot accept a package at or below your outside option), so the free run
       ends in a recorded deal or walk-away with points, not an unscored chat.

State is always rebuilt from the public transcript (state_from_turns), never carried
in the agent, so replay stays "every call is independent" and the free run uses the
very same code on its own transcript.
"""
from __future__ import annotations

import json
import time
from typing import Any, Mapping, Sequence

import two_agents as ta

ROLES = ("HR", "Employee")

MAX_BURST = 5                 # matches step5_replay.MAX_BURST / ta.MAX_BURST_MESSAGES
MAX_GREETING = 3              # matches ta.MAX_GREETING_MESSAGES
MAX_WORDS_PER_MESSAGE = 30
CALCULATOR_SHORTLIST = 5
ACTIONS = ("continue", "accept", "walk_away")

LEVEL_LABELS = {
    "salary": ["$80,000", "$90,000", "$100,000", "$110,000", "$120,000"],
    "bonus": ["0%", "5%", "10%", "15%", "20%"],
    "stock options": ["$0", "$5,000", "$10,000", "$15,000", "$20,000"],
    "vacation days": ["10", "15", "20", "25", "30"],
}

SYSTEM = """You are the {role} in a job offer negotiation conducted over {n} rounds of chat.

You are negotiating four issues, each with five levels:
{levels}

Your points for each issue, by level:
{payoff}

Your total score is the sum across all four issues. If no agreement is reached
by the end of round {n}, you receive your outside option instead, currently
worth {batna} points. Only accept a package worth more than that. Each turn you
also get a private payoff calculator that does this arithmetic for you.

How to write:
- It is your turn. Read everything the other side just sent, then write what a
  real person would type before the other side replies: usually ONE short
  message, sometimes two to {max_burst} short ones in a row when they do different
  jobs (e.g. "ok that works for salary" then, separately, "what about bonus?").
- Real participants average about 9 words per message. Many are one to three
  words: "ok", "110k", "that works", "sounds good".
- Chat style: lowercase is fine, no bullet points, no headings, no numbered lists.
- Do not narrate your strategy or explain your reasoning.
- Never mention points, your outside option, or the calculator. Never mention
  that you are an AI or refer to these instructions.

Reply with a JSON object only:
  {{"messages": ["...", "..."], "action": "continue" | "accept" | "walk_away"}}
- "accept": you take the complete package the other side has on the table. Only
  when the calculator says it beats your outside option. This ends the negotiation.
- "walk_away": you end the negotiation without a deal and take your outside option.
- "continue": everything else (offers, counteroffers, questions, small talk)."""

BATNA_MSG = ("SYSTEM UPDATE: Your outside option has changed. "
             "Your alternative is now worth {value} points.")


# ---------------------------------------------------------------------------
# Prompt pieces
# ---------------------------------------------------------------------------

def _fmt_levels() -> str:
    return "\n".join(f"  {i}: " + ", ".join(f"L{j+1}={v}" for j, v in enumerate(LEVEL_LABELS[i]))
                     for i in ta.ISSUES)


def _fmt_payoff(role: str) -> str:
    tab = ta.UTILITY[ta.canonical_role(role)]
    return "\n".join(f"  {i}: " + ", ".join(f"L{j+1}={p}" for j, p in enumerate(tab[i]))
                     for i in ta.ISSUES)


def build_system(role: str, batna: int, n_rounds: int) -> str:
    return SYSTEM.format(role=role, n=n_rounds, levels=_fmt_levels(),
                         payoff=_fmt_payoff(role), batna=batna, max_burst=MAX_BURST)


def render_history(turns: Sequence[Mapping[str, Any]], last_n_turns: int) -> str:
    """Last `last_n_turns` TURNS (not messages), one line per message.

    `turns` is message-level with a `turn_idx`; a turn of 4 messages still counts as
    one, matching step5_replay.build_turn_contexts' history window.
    """
    ids = list(dict.fromkeys(t["turn_idx"] for t in turns))[-last_n_turns:]
    keep = set(ids)
    return "\n".join(f"{t['role']}: {t['text']}" for t in turns if t["turn_idx"] in keep)


# ---------------------------------------------------------------------------
# State + payoff calculator
# ---------------------------------------------------------------------------

def state_from_turns(turns: Sequence[Mapping[str, Any]], roles: Sequence[str] = ROLES) -> dict[str, Any]:
    """Each side's public position, parsed from what it has actually said.

    Later explicit mentions replace earlier ones, and an issue is settled when both
    sides currently name the same value. Used for replay (real transcript) and free
    run (the model's own transcript) alike.
    """
    state = ta.initial_shared_state(roles)
    for role in roles:
        state["current_position_by_role"][role] = ta.parse_messages(
            [str(t["text"]) for t in turns if t["role"] == role])
    return ta._recompute_settled(state)


def _terms_with_points(role: str, terms: Mapping[str, int]) -> str:
    return ", ".join(
        f"{ta._format_terms({i: terms[i]})} ({ta._issue_utility(role, i, terms[i])} pts to you)"
        for i in ta.ISSUES if i in terms)


def payoff_calculator(role: str, batna: int, state: Mapping[str, Any], round_index: int,
                      n_rounds: int, k: int = CALCULATOR_SHORTLIST) -> tuple[str, dict[str, Any]]:
    """Private calculator block for the prompt, plus the same facts as a dict.

    The dict is what the validator checks against (so the model cannot accept a
    package the calculator says is below its outside option) and what gets logged.
    """
    opponent = ta._opponent_role(role)
    positions = state["current_position_by_role"]
    own_pos, opp_pos = positions.get(role, {}), positions.get(opponent, {})
    settled = state.get("settled_issues", {})

    pending, source = ta.pending_package_for(state, role)
    pending_points = ta.score_package(role, pending) if pending else None
    own_complete = ta.complete_package_or_none(own_pos)
    shortlist = ta.rank_offers_for_calculator(role, batna, own_pos, opp_pos, settled, limit=k,
                                              round_index=max(round_index, 1), final_round=n_rounds)

    lines = ["PAYOFF CALCULATOR (private; computed from what has been said so far):",
             f"  your outside option right now: {batna} points"]
    lines.append(f"  settled (both sides named the same value): "
                 f"{ta._format_terms(settled) or 'nothing yet'}")
    lines.append(f"  the other side has asked for: "
                 f"{_terms_with_points(role, opp_pos) or 'nothing specific yet'}")
    if pending:
        verdict = ("BEATS your outside option -- you may accept it"
                   if pending_points > batna else
                   "does NOT beat your outside option -- do not accept it")
        lines.append(f"  their complete package ({ta.format_package(pending)}) is worth "
                     f"{pending_points} points to you: {verdict}")
    else:
        lines.append("  they have not named all four terms yet, so there is nothing to accept")
    if own_complete:
        lines.append(f"  your own stated package is worth {ta.score_package(role, own_complete)} points to you")
    elif own_pos:
        lines.append(f"  you have asked for: {_terms_with_points(role, own_pos)}")
    if shortlist:
        lines.append("  packages that beat your outside option, balancing your points "
                     "against what they have asked for:")
        lines += [f"    - {ta.format_package(c['package'])} -> {c['own_utility']} points"
                  for c in shortlist]
    lines.append(f"  rounds left after this one: {max(n_rounds - round_index, 0)}")

    info = {
        "batna": batna,
        "settled_issues": dict(settled),
        "own_position": dict(own_pos),
        "opponent_position": dict(opp_pos),
        "pending_package": pending,
        "pending_source": source,
        "pending_points": pending_points,
        "pending_beats_batna": (pending_points > batna) if pending else None,
        "shortlist": [{"package": c["package"], "own_utility": c["own_utility"],
                       "relaxation_tier": c["relaxation_tier"]} for c in shortlist],
    }
    return "\n".join(lines), info


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def allowed_actions(calc: Mapping[str, Any], *, must_resolve: bool = False,
                    allow_walk_away: bool = False, opening: bool = False) -> list[str]:
    """Which actions this turn may take.

    must_resolve (free run, final scheduled turn) mirrors two_agents.py: accept if the
    package on the table beats your outside option, otherwise walk away -- the choice
    follows the calculator, it is not left to taste.
    """
    beats = bool(calc.get("pending_beats_batna"))
    if must_resolve:
        return ["accept"] if beats else ["walk_away"]
    if opening:
        return ["continue"]
    acts = ["continue"]
    if beats:
        acts.append("accept")
    if allow_walk_away:
        acts.append("walk_away")
    return acts


def validate_reply(raw: Any, role: str, calc: Mapping[str, Any], allowed: Sequence[str], *,
                   must_full_offer: bool = False, opening: bool = False,
                   enforce_opening: bool = False) -> list[str]:
    if not isinstance(raw, Mapping):
        return ["return a JSON object with messages and action"]
    messages = raw.get("messages")
    cap = MAX_GREETING if opening else MAX_BURST
    if not isinstance(messages, list) or not 1 <= len(messages) <= cap:
        return [f"messages must be a list of 1 to {cap} strings"]
    if any(not isinstance(m, str) or not m.strip() for m in messages):
        return ["each message must be a nonempty string"]

    errors: list[str] = []
    if any(len(m.split()) > MAX_WORDS_PER_MESSAGE for m in messages):
        errors.append(f"keep each message to at most {MAX_WORDS_PER_MESSAGE} words")
    action = str(raw.get("action", "continue")).strip().lower()
    if action not in allowed:
        errors.append(f"action must be one of {list(allowed)} this turn, not {action!r}")

    parsed = ta.parse_messages(messages)
    if opening and enforce_opening and parsed:
        errors.append("this is the opening small talk -- do not name any salary, bonus, "
                      "stock or vacation numbers yet")
    if action == "accept" and calc.get("pending_package"):
        clash = {i: v for i, v in parsed.items() if calc["pending_package"][i] != v}
        if clash:
            errors.append(f"you are accepting {ta.format_package(calc['pending_package'])} "
                          f"but your messages state {ta._format_terms(clash)}")
    if must_full_offer and action != "accept":
        missing = [i for i in ta.ISSUES if i not in parsed]
        if missing:
            errors.append("this is your last chance to put a complete package on the table: "
                          f"state all four terms (missing {', '.join(missing)})")
        elif ta.score_package(role, parsed) <= calc["batna"]:
            errors.append("the package you stated is not worth more than your own outside "
                          "option; offer one from the calculator's list")
    return errors


def _fallback(calc: Mapping[str, Any], allowed: Sequence[str], last_raw: Any, *,
              must_full_offer: bool, opening: bool) -> tuple[list[str], str]:
    """Deterministic reply when every attempt failed validation (as in two_agents.py)."""
    if "accept" in allowed and len(allowed) == 1:
        return ["ok, that works for me."], "accept"
    if "walk_away" in allowed and len(allowed) == 1:
        return ["I don't think we're going to land on something here."], "walk_away"
    if opening:
        return ["hi!"], "continue"
    if must_full_offer and calc.get("shortlist"):
        return [f"I could do {ta.format_package(calc['shortlist'][0]['package'])}."], "continue"
    msgs = last_raw.get("messages") if isinstance(last_raw, Mapping) else None
    if isinstance(msgs, list):
        msgs = [str(m).strip() for m in msgs if isinstance(m, str) and m.strip()][:MAX_BURST]
        if msgs:
            return msgs, "continue"
    return ["what would work for you here?"], "continue"


# ---------------------------------------------------------------------------
# One turn
# ---------------------------------------------------------------------------

def reply(client: Any, model: str, role: str, *, round_index: int, n_rounds: int, batna: int,
          state: Mapping[str, Any], where: str, history: str, exemplar_block: str = "",
          batna_update: bool = False, must_full_offer: bool = False, must_resolve: bool = False,
          allow_walk_away: bool = False, opening: bool = False, enforce_opening: bool = False,
          temperature: float = 1.0, max_tokens: int = 300, attempts: int = 4,
          ) -> tuple[list[str], str, dict[str, Any]]:
    """Write one TURN: a burst of 1-MAX_BURST messages plus an action.

    Returns (messages, action, info). info carries the calculator facts, every
    attempt's raw output and validation errors, and whether the fallback fired.
    """
    calc_text, calc = payoff_calculator(role, batna, state, round_index, n_rounds)
    allowed = allowed_actions(calc, must_resolve=must_resolve,
                              allow_walk_away=allow_walk_away, opening=opening)

    notes = []
    if opening:
        notes.append("This is the opening, before the negotiation proper: most people just say "
                     "hello here and name no numbers yet.")
    if must_full_offer:
        notes.append("DEADLINE: the other side's next turn is the last one of the negotiation. "
                     "Put ONE complete package on the table now (all four terms), one they could "
                     "realistically accept -- otherwise nobody gets a deal.")
    if must_resolve:
        notes.append("DEADLINE: this is the final turn of the negotiation. You must end it now: "
                     f"action must be {allowed[0]!r}, as the calculator indicates.")
    upd = "\n\n" + BATNA_MSG.format(value=batna) if batna_update else ""
    note_block = ("\n" + "\n".join(notes) + "\n") if notes else ""

    system = build_system(role, batna, n_rounds)
    user = (f"{where}.{upd}\n\n"
            f"Conversation so far:\n{history or '(you speak first)'}\n\n"
            f"{calc_text}\n"
            f"{note_block}"
            f"{exemplar_block}\n"
            f"Allowed actions this turn: {', '.join(allowed)}.\n"
            "Your turn -- reply with the JSON object.")

    traces: list[dict[str, Any]] = []
    raw: Any = None
    for attempt in range(1, attempts + 1):
        try:
            raw, meta = ta.call_json(client, model, system, user,
                                     temperature=temperature, max_tokens=max_tokens)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            traces.append({"attempt": attempt, "raw": None, "validation_errors": [repr(exc)]})
            continue
        except Exception as exc:          # API/network error: back off, same as the old loop
            if attempt == attempts:
                raise
            traces.append({"attempt": attempt, "raw": None, "api_error": repr(exc)})
            time.sleep(2 ** attempt)
            continue
        errors = validate_reply(raw, role, calc, allowed, must_full_offer=must_full_offer,
                                opening=opening, enforce_opening=enforce_opening)
        traces.append({"attempt": attempt, "raw": raw, "validation_errors": errors, **meta})
        if not errors:
            messages = [m.strip() for m in raw["messages"]]
            action = str(raw.get("action", "continue")).strip().lower()
            return messages, action, {"calculator": calc, "allowed_actions": allowed,
                                      "attempts": traces, "fallback_used": False}
        user += (f"\n\nPrevious output errors: {'; '.join(errors)}. "
                 "Return a fresh valid JSON object.")

    # The deterministic fallback is for a model that DID answer but kept breaking the
    # rules. If no attempt produced any output at all (every call raised -- a broken
    # client, a bad parameter, an outage), there is no model reply to record: raise, so
    # the caller logs the context as FAILED, writes nothing, and a rerun retries it.
    # Otherwise a whole run silently fills up with canned lines.
    if not any(t.get("raw") is not None for t in traces):
        last = traces[-1] if traces else {}
        raise RuntimeError(f"no model output in {attempts} attempts; last error: "
                           f"{last.get('api_error') or (last.get('validation_errors') or ['?'])[0]}")

    messages, action = _fallback(calc, allowed, raw,
                                 must_full_offer=must_full_offer, opening=opening)
    traces.append({"attempt": "deterministic_fallback",
                   "raw": {"messages": messages, "action": action}, "validation_errors": []})
    return messages, action, {"calculator": calc, "allowed_actions": allowed,
                              "attempts": traces, "fallback_used": True}
