"""two_agents.py — a simpler two-agent negotiation architecture.

Each negotiating side owns two persistent agents instead of three:

1) StrategyPlannerAgent: each turn it does two things in one call — reads what
   the opponent's last public move actually signals, then decides what to do
   next. It has access to an offer calculator (rank_offers) that enumerates
   every legal package, throws out anything at or below BATNA, anything that
   silently reverses a concession already made, and anything that contradicts
   an issue both sides have already converged on, then ranks the survivors by
   the planner's own payoff and by how close they sit to what the opponent has
   actually asked for. The planner picks off that top-12 shortlist (or asks,
   holds, accepts, or walks away) — it never invents an off-grid number.

2) LanguageRealizerAgent: turns the planner's decision into one or two short,
   casual chat messages. It never sees private payoff/BATNA information, only
   the decision, the recent conversation, and retrieved human examples.

WHY THIS IS SIMPLER THAN three_agents.py
    three_agents.py splits opponent-modeling out into a third agent that
    produces a structured, LLM-inferred preference object, which the offer
    calculator then uses to score candidates before the strategist ever runs.
    Here the calculator scores candidates arithmetically against what the
    opponent has actually and publicly said (no separate inference call, no
    preference object to validate), and the one planner call does the
    judgment a person would: read the room, then move. Structural validation
    is limited to what actually protects the game — legal grid values, the
    BATNA floor, and not silently contradicting a settled issue — rather than
    the much larger rulebook in three_agents.py (tone/meta-language
    detectors, firm-term bookkeeping, burst redundancy checks, ...). The rest
    is left to the prompt and to the state the agents are shown.

CONSISTENCY ACROSS ROUNDS
    settled_issues is recomputed from public state every turn: whenever both
    sides are currently stating the same value for an issue, it's settled,
    and the offer calculator locks it there for every future shortlist — so
    a party literally cannot be handed a candidate package that reopens
    salary once salary is settled. A planner may still explicitly reopen a
    settled issue (reopen_issues), but that's a deliberate, visible move, not
    a default. Since settled_issues is derived from state rather than a
    ratchet flag, it also un-settles itself the moment either side actually
    moves off it — no separate bookkeeping to keep in sync.

HUMAN TONE CALIBRATION
    The realizer's brevity/casualness targets below (aim ~6-12 words per
    message, treat bare fragments as normal, contractions and lowercase are
    fine) come from Raiyan_analysis_new/data/updated_dataset.csv: mean 9.1
    words per human message, ~33% of messages at or under 4 words, ~13% use
    a contraction, and ~31% start lowercase. Real "settle and move on" turns
    in that data read like "so we agree on vac days, are you flexible on
    anything else" — plain, short, no meta-language — which is what the
    realizer's system prompt is aiming for, not a corporate-sounding recap.

INTERFACE
    Mirrors three_agents.py / multi_agent.py so a new condition prefix (e.g.
    "2A-") can reuse step2_generate.py's run_multi_agent the same way MA-*
    and 3A-* do today:

        simulate_multi_agent_dyad(client, game_id, roles, rng, batna_target,
                                   retrieve_examples, config) -> (public, traces)

    retrieve_examples(role, round_index, phase, behavior, history)
        -> (examples: list[str], example_ids: list[Any])

    The signature matches three_agents.py's retrieve_examples exactly, but the
    4th argument here is the CURRENT turn's behavior category (Neutral /
    Integrative / Distributive, from the planner's already-chosen action),
    not the previous turn's -- the planner runs before retrieval, so the
    actual category is known and there's no gap on a role's first turn.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Mapping, Sequence

# =============================================================================
# 1. THE GAME (identical grid/payoffs to three_agents.py / multi_agent.py, so
#    results stay comparable across architectures)
# =============================================================================

ISSUES = ("salary", "bonus", "stock options", "vacation days")
LEVEL_VALUES = {
    "salary": (80000, 90000, 100000, 110000, 120000),
    "bonus": (0, 5, 10, 15, 20),
    "stock options": (0, 5000, 10000, 15000, 20000),
    "vacation days": (10, 15, 20, 25, 30),
}
UTILITY = {
    "HR": {
        "salary": (15, 12, 9, 6, 3),
        "bonus": (15, 12, 9, 6, 3),
        "stock options": (-1, 0, 1, 2, 3),
        "vacation days": (30, 24, 18, 12, 6),
    },
    "Employee": {
        "salary": (6, 12, 18, 24, 30),
        "bonus": (3, 6, 9, 12, 15),
        "stock options": (2, 4, 6, 8, 10),
        "vacation days": (3, 6, 9, 12, 15),
    },
}
UTILITY_RANGE = {"HR": (11, 63), "Employee": (14, 70)}

ACTIONS = {"offer", "accept", "ask", "hold", "walk_away", "greet"}
TERMINAL_ACTIONS = {"accept", "walk_away"}
# Matches step5_replay.MAX_BURST -- the realizer may send this many chat messages for
# one strategic decision, mirroring how many consecutive messages real humans send
# before the other side replies (replaces the old react_first-forced-2-message rule).
MAX_BURST_MESSAGES = 5
# Pure small-talk bursts run shorter than negotiation bursts in the real data (139/155
# all-Neutral turns are 1 message, 11 are 2, 2 are 3, a couple of rare outliers beyond
# that) -- e.g. "Hi, Dan" / "Hello" / "Are you there?" as three separate lines before
# the other side replies. 3 covers 98% of that without inviting greeting babble.
MAX_GREETING_MESSAGES = 3

Package = dict[str, int]
TWO_AGENT_VERSION = "0.1.0-planner-realizer"


def canonical_role(role: str) -> str:
    aliases = {"hr": "HR", "employee": "Employee", "candidate": "Employee"}
    try:
        return aliases[role.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported role: {role!r}") from exc


def _opponent_role(role: str) -> str:
    return next(r for r in ("HR", "Employee") if r != canonical_role(role))


def validate_package(package: Mapping[str, int]) -> Package:
    if set(package) != set(ISSUES):
        raise ValueError(f"package must contain exactly: {', '.join(ISSUES)}")
    clean = {issue: int(package[issue]) for issue in ISSUES}
    for issue, value in clean.items():
        if value not in LEVEL_VALUES[issue]:
            raise ValueError(f"off-grid {issue}: {value}")
    return clean


def complete_package_or_none(value: Any) -> Package | None:
    if not isinstance(value, Mapping) or set(value) != set(ISSUES):
        return None
    try:
        return validate_package(value)
    except (TypeError, ValueError):
        return None


def package_levels(package: Mapping[str, int]) -> dict[str, int]:
    clean = validate_package(package)
    return {issue: LEVEL_VALUES[issue].index(clean[issue]) + 1 for issue in ISSUES}


def _levels(partial: Mapping[str, int]) -> dict[str, int]:
    """Like package_levels but tolerates a partial (not necessarily complete) mapping."""
    return {issue: LEVEL_VALUES[issue].index(value) + 1 for issue, value in partial.items()}


def score_package(role: str, package: Mapping[str, int]) -> int:
    role = canonical_role(role)
    levels = package_levels(package)
    return sum(UTILITY[role][issue][level - 1] for issue, level in levels.items())


def normalized_utility(role: str, package: Mapping[str, int]) -> float:
    role = canonical_role(role)
    low, high = UTILITY_RANGE[role]
    return (score_package(role, package) - low) / (high - low)


def _issue_utility(role: str, issue: str, value: int) -> int:
    level = LEVEL_VALUES[issue].index(value)
    return UTILITY[canonical_role(role)][issue][level]


def enumerate_packages():
    for values in product(*(LEVEL_VALUES[issue] for issue in ISSUES)):
        yield dict(zip(ISSUES, values))


def _format_terms(terms: Mapping[str, int]) -> str:
    labels = {
        "salary": lambda v: f"{v // 1000}k salary",
        "bonus": lambda v: f"{v}% bonus",
        "stock options": lambda v: f"{v // 1000}k stock",
        "vacation days": lambda v: f"{v} vacation days",
    }
    return ", ".join(labels[issue](terms[issue]) for issue in ISSUES if issue in terms)


def format_package(package: Mapping[str, int]) -> str:
    return _format_terms(validate_package(package))


def format_partial_package(package: Mapping[str, int], issues: Sequence[str]) -> str:
    p = validate_package(package)
    return _format_terms({issue: p[issue] for issue in issues})


def format_human_style_examples(examples: Sequence[str]) -> str:
    """Render retrieved human lines as a salient prose block, not a buried JSON array."""
    quoted = [str(example).strip() for example in examples if str(example).strip()]
    if not quoted:
        return ""
    lines = "\n".join(f'  "{example}"' for example in quoted)
    return (
        "\n\nHow real participants wrote at similar moments:\n"
        f"{lines}\n"
        "Match that tone and length. Do NOT reuse their wording; write your own message."
    )


# =============================================================================
# 2. PARSING PUBLIC TEXT (extract exact on-grid values so state tracks what
#    was actually said, not just what a decision object claims)
# =============================================================================

# A bare, unit-free package summary in canonical salary/bonus/stock/vacation order --
# e.g. "I could do 110, 20, 20, 10" -- real negotiators use this near the end of a
# negotiation, exactly where getting state right matters most. Only tried when the
# labeled-field regexes below find nothing at all, and only accepted if ALL FOUR
# numbers land on their issue's legal grid value -- that's what keeps this from
# misreading an unrelated number run as a package (a random 4-tuple landing on 4
# separate 5-value grids simultaneously is unlikely; verified against the corpus:
# 4 messages match the shape, 3 are real packages, 1 -- "120, 10,1000,15" -- is
# correctly rejected because 1000 isn't a legal stock-options value).
_FOUR_BARE_NUMBERS = re.compile(r"(?<!\d)(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?!\d)")


def _bare_package_shorthand(text: str) -> dict[str, int] | None:
    m = _FOUR_BARE_NUMBERS.search(text)
    if not m:
        return None
    raw = [int(x) for x in m.groups()]
    candidate = {
        "salary": raw[0] * 1000 if raw[0] < 1000 else raw[0],
        "bonus": raw[1],
        "stock options": raw[2] * 1000 if raw[2] < 1000 else raw[2],
        "vacation days": raw[3],
    }
    return candidate if all(v in LEVEL_VALUES[i] for i, v in candidate.items()) else None


def parse_offer(text: str) -> dict[str, int]:
    """Extract exact on-grid issue values from a natural-language message."""
    lower = text.lower()
    result: dict[str, int] = {}

    salary_matches = re.findall(r"\$?\s*(80|90|100|110|120)(?:[,.]?000|k)\b", lower)
    if salary_matches:
        result["salary"] = int(salary_matches[0]) * 1000

    bonus = re.search(r"\b(0|5|10|15|20)\s*%", lower)
    if bonus:
        result["bonus"] = int(bonus.group(1))
    elif re.search(r"\bno\s+bonus\b", lower):
        result["bonus"] = 0
    else:
        bonus_value = re.search(r"\bbonus\s*(?:of|at|:|=)?\s*\$?\s*(0|5|10|15|20)\b", lower)
        if bonus_value:
            result["bonus"] = int(bonus_value.group(1))

    stock = re.search(
        r"\$?\s*(0|5|10|15|20)(?:[,.]?000|k)\s*(?:in\s+)?(?:stock|options?)", lower,
    ) or re.search(
        r"(?:stock(?:\s+options?)?|options?)\s*(?:of|at|:|=)?\s*\$?\s*(0|5|10|15|20)(?:[,.]?000|k)?\b", lower,
    )
    if stock:
        value = int(stock.group(1))
        result["stock options"] = value * 1000 if value else 0

    vacation = re.search(
        r"\b(10|15|20|25|30)\s*(?:vacation\s*)?(?:days?|days?\s+off|vaca(?:tion)?)\b", lower,
    ) or re.search(r"(?:vacation|vaca)[^\d]{0,18}\b(10|15|20|25|30)\b", lower)
    if vacation:
        result["vacation days"] = int(vacation.group(1))

    result = {issue: value for issue, value in result.items() if value in LEVEL_VALUES[issue]}
    if result:
        return result
    return _bare_package_shorthand(text) or {}


def parse_messages(messages: Sequence[str]) -> dict[str, int]:
    """Parse a burst in order; later explicit mentions replace earlier ones."""
    parsed: dict[str, int] = {}
    for message in messages:
        parsed.update(parse_offer(str(message)))
    return parsed


# =============================================================================
# 3. SHARED STATE — everything both agents (and the offer calculator) can see
# =============================================================================

def initial_shared_state(roles: Sequence[str]) -> dict[str, Any]:
    return {
        "current_position_by_role": {role: {} for role in roles},
        "last_complete_package_by_role": {role: None for role in roles},
        "outstanding_offer": None,
        "settled_issues": {},
        "deal": None,
        "asked_about_by_role": {role: [] for role in roles},
        "last_action_by_role": {role: None for role in roles},
        "batna_updates": {role: False for role in roles},
    }


def _recompute_settled(state: dict[str, Any]) -> dict[str, Any]:
    positions = state["current_position_by_role"]
    roles = list(positions)
    if len(roles) != 2:
        return state
    a, b = positions[roles[0]], positions[roles[1]]
    state["settled_issues"] = {
        issue: a[issue] for issue in ISSUES if issue in a and issue in b and a[issue] == b[issue]
    }
    return state


def pending_package_for(state: Mapping[str, Any], role: str) -> tuple[Package | None, str]:
    """The complete opponent package this side could accept right now, and where it came from.

    Two sources, in priority order:

      "offer"    the opponent's outstanding_offer, when it was a full-scope offer.
      "position" the opponent's accumulated current_position_by_role, once it names
                 all four issues.

    The second source exists because real dyads close on the accumulated position,
    not on a formally complete package -- "ok i think i can do that exactly too"
    accepts four numbers that arrived across several partial offers. Recognising
    only full-scope offers meant a side that had stated every number through
    partials presented nothing acceptable, so its counterpart was forced to walk
    away with own_utility=None. Both sources are still gated by the BATNA rule at
    the call sites; this helper only says what is on the table.
    """
    opponent = _opponent_role(role)
    outstanding = state.get("outstanding_offer") or {}
    if outstanding.get("offered_by") not in (None, role):
        package = complete_package_or_none(outstanding.get("package"))
        if package:
            return package, "offer"
    positions = state.get("current_position_by_role", {}) or {}
    package = complete_package_or_none(positions.get(opponent, {}))
    if package:
        return package, "position"
    return None, "none"


def update_shared_state(
    state: Mapping[str, Any], role: str, decision: Mapping[str, Any], messages: Sequence[str]
) -> dict[str, Any]:
    updated = json.loads(json.dumps(state))
    action = decision["action"]
    positions = updated["current_position_by_role"]

    if action == "offer":
        terms = dict(decision.get("terms") or {})
        positions[role] = {**positions.get(role, {}), **terms}
        complete = complete_package_or_none(positions[role])
        if complete:
            updated["last_complete_package_by_role"][role] = complete
        updated["outstanding_offer"] = {
            "offered_by": role,
            "scope": decision.get("scope"),
            "package": complete_package_or_none(decision.get("package")),
            "terms": terms,
        }
    elif action == "accept":
        outstanding = updated.get("outstanding_offer") or {}
        package = outstanding.get("package")
        updated["deal"] = package
        if package:
            positions[role] = {**positions.get(role, {}), **package}
        updated["outstanding_offer"] = None
    elif action in ("ask", "hold"):
        pass
    elif action == "walk_away":
        updated["outstanding_offer"] = None

    updated["last_action_by_role"][role] = action
    if action == "ask":
        asked = updated["asked_about_by_role"].setdefault(role, [])
        for issue in decision.get("focus_issues") or []:
            if issue not in asked:
                asked.append(issue)

    return _recompute_settled(updated)


# =============================================================================
# 4. THE OFFER CALCULATOR — arithmetic ranking, no LLM call
# =============================================================================

def _is_reversal(own_position: Mapping[str, int], candidate: Mapping[str, int], role: str) -> bool:
    """True if `candidate` quietly gives the opponent less than we've already offered them."""
    opponent = _opponent_role(role)
    for issue, value in own_position.items():
        if issue not in candidate:
            continue
        if _issue_utility(opponent, issue, candidate[issue]) < _issue_utility(opponent, issue, value):
            return True
    return False


def _proximity_to_opponent(candidate: Mapping[str, int], opponent_position: Mapping[str, int]) -> float:
    known = [issue for issue in ISSUES if issue in opponent_position]
    if not known:
        return 0.5
    cand, opp = _levels(candidate), _levels(opponent_position)
    return sum(1.0 - abs(cand[i] - opp[i]) / 4 for i in known) / len(known)


def _continuity_distance(candidate: Mapping[str, int], own_position: Mapping[str, int]) -> int:
    known = [issue for issue in ISSUES if issue in own_position]
    if not known:
        return 0
    cand, own = _levels(candidate), _levels(own_position)
    return sum(abs(cand[i] - own[i]) for i in known)


def rank_offers(
    role: str,
    batna: int,
    own_position: Mapping[str, int],
    opponent_position: Mapping[str, int],
    locked_terms: Mapping[str, int],
    limit: int = 12,
    preserve_concessions: bool = True,
) -> list[dict[str, Any]]:
    """Enumerate legal packages and rank the ones worth showing the planner.

    Ranking balances the planner's own payoff against how close a candidate
    sits to what the opponent has actually, publicly asked for — an
    arithmetic stand-in for "what the opponent wants" that needs no separate
    preference-inference call. `locked_terms` (normally state["settled_issues"])
    is enforced structurally: no candidate can vary a locked issue.
    """
    own_position = dict(own_position or {})
    opponent_position = dict(opponent_position or {})
    locked_terms = dict(locked_terms or {})
    current_complete = complete_package_or_none(own_position)

    candidates = []
    for package in enumerate_packages():
        if any(package[issue] != value for issue, value in locked_terms.items()):
            continue
        if current_complete is not None and package == current_complete:
            continue
        own = score_package(role, package)
        if own <= batna:
            continue
        if preserve_concessions and _is_reversal(own_position, package, role):
            continue
        candidates.append({
            "package": package,
            "own_utility": own,
            "own_normalized": round(normalized_utility(role, package), 6),
            "opponent_proximity": round(_proximity_to_opponent(package, opponent_position), 6),
            "continuity_distance": _continuity_distance(package, own_position),
        })

    for candidate in candidates:
        candidate["strategic_score"] = round(
            0.7 * candidate["own_normalized"]
            + 0.3 * candidate["opponent_proximity"]
            - min(candidate["continuity_distance"], 8) * 0.01,
            6,
        )
    candidates.sort(key=lambda c: (
        -c["strategic_score"], -c["own_utility"], c["continuity_distance"],
        tuple(c["package"][issue] for issue in ISSUES),
    ))
    shortlist = candidates[:limit]
    for index, candidate in enumerate(shortlist, start=1):
        candidate["candidate_id"] = f"offer_{index:02d}"
    return shortlist


def _concession_weight(round_index: int | None, final_round: int) -> float:
    """How much a candidate's own payoff outweighs closeness to the opponent.

    Constant 0.7 through every round was why sides arrived at the deadline still
    holding near-greedy packages: the calculator applied no time pressure, so the
    penultimate full package was routinely one the other side had to reject. Real
    negotiators concede on a deadline, so the own-payoff weight decays from 0.7 in
    the opening round to 0.4 in the final one, which tilts the shortlist toward
    what the opponent has actually asked for as the clock runs out. Passing
    round_index=None keeps the historical flat 0.7 (rank_offers()'s own behavior,
    and every caller that has no round).
    """
    if round_index is None or final_round <= 1:
        return 0.7
    progress = min(max((round_index - 1) / (final_round - 1), 0.0), 1.0)
    return 0.7 - 0.3 * progress


def _rank_offers_tier(
    role: str,
    batna: int,
    own_position: Mapping[str, int],
    opponent_position: Mapping[str, int],
    locked_terms: Mapping[str, int],
    limit: int,
    preserve_concessions: bool,
    own_weight: float,
) -> list[dict[str, Any]]:
    """One pass of the calculator under a fixed set of filters. See rank_offers_for_calculator()."""
    current_complete = complete_package_or_none(own_position)
    candidates = []
    for package in enumerate_packages():
        if any(package[issue] != value for issue, value in locked_terms.items()):
            continue
        if current_complete is not None and package == current_complete:
            continue
        own = score_package(role, package)
        # The BATNA floor is never relaxed at any tier: a side must not table an
        # offer it would itself be required to reject.
        if own <= batna:
            continue
        if preserve_concessions and _is_reversal(own_position, package, role):
            continue
        candidates.append({
            "package": package,
            "own_utility": own,
            "own_normalized": round(normalized_utility(role, package), 6),
            "opponent_proximity": round(_proximity_to_opponent(package, opponent_position), 6),
            "continuity_distance": _continuity_distance(package, own_position),
        })

    for candidate in candidates:
        candidate["strategic_score"] = round(
            own_weight * candidate["own_normalized"]
            + (1.0 - own_weight) * candidate["opponent_proximity"]
            - min(candidate["continuity_distance"], 8) * 0.01,
            6,
        )
    candidates.sort(key=lambda c: (
        -c["strategic_score"], -c["own_utility"], c["continuity_distance"],
        tuple(c["package"][issue] for issue in ISSUES),
    ))
    return candidates[:limit]


def rank_offers_for_calculator(
    role: str,
    batna: int,
    own_position: Mapping[str, int],
    opponent_position: Mapping[str, int],
    locked_terms: Mapping[str, int],
    limit: int = 12,
    preserve_concessions: bool = True,
    round_index: int | None = None,
    final_round: int = 6,
) -> list[dict[str, Any]]:
    """Like rank_offers(), plus deadline-aware concession weighting and graduated
    relaxation -- used by single_agent.py's payoff_calculator(), kept as its own
    function (not a change to rank_offers() itself) so the multi-agent planner's
    existing behavior stays exactly as it was.

    GRADUATED RELAXATION — why this is tiered
        The strict filter set (settled issues locked, no reversals, must beat BATNA)
        can and did return an EMPTY shortlist: once three of four issues were settled
        and a side had conceded down to just above its BATNA, no legal package
        remained, so the side had no offer it was permitted to make. That deadlock was
        the direct cause of an all-walk-away pilot run — the penultimate speaker could
        not table a complete package, so the final speaker had nothing to accept.

        So the calculator falls back one constraint at a time instead of returning
        nothing, and reports which tier it used:

          0  strict         — settled locked, concessions preserved
          1  reversals      — settled still locked, reversals permitted
          2  reopen         — settled unlocked too (candidates tagged `reopens`)

        The BATNA floor is never relaxed; it is the protocol's rule, not a heuristic.
        Every candidate carries `relaxation_tier`, and tier-2 candidates carry the
        `reopens` list, so a reasoner choosing one knows it must list those issues in
        reopen_issues and the validator still holds it to that.
    """
    own_position = dict(own_position or {})
    opponent_position = dict(opponent_position or {})
    locked_terms = dict(locked_terms or {})
    own_weight = _concession_weight(round_index, final_round)

    tiers = (
        (0, locked_terms, preserve_concessions),
        (1, locked_terms, False),
        (2, {}, False),
    )
    shortlist: list[dict[str, Any]] = []
    tier_used = 0
    for tier, locks, preserve in tiers:
        if tier > 0 and not (locks != locked_terms or preserve != preserve_concessions):
            continue  # this tier relaxes nothing the caller had not already relaxed
        shortlist = _rank_offers_tier(
            role, batna, own_position, opponent_position, locks, limit, preserve, own_weight,
        )
        tier_used = tier
        if shortlist:
            break

    for index, candidate in enumerate(shortlist, start=1):
        candidate["candidate_id"] = f"offer_{index:02d}"
        candidate["relaxation_tier"] = tier_used
        candidate["reopens"] = sorted(
            issue for issue, value in locked_terms.items() if candidate["package"][issue] != value
        )
    return shortlist


# =============================================================================
# 5. LLM CALL HELPER
# =============================================================================

def _response_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, Mapping):
        return dict(usage)
    return {k: getattr(usage, k) for k in ("prompt_tokens", "completion_tokens", "total_tokens") if hasattr(usage, k)}


def call_json(
    client: Any, model: str, system: str, user: str, temperature: float = 0.2, max_tokens: int = 500,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    response = client.chat.completions.create(
        # client comes from llm_backends.make_client(), whose shims all take `max_tokens`
        # and translate per-provider internally (e.g. gpt-5.4 needs max_completion_tokens
        # plus a reasoning-token headroom, handled inside the shim, not here).
        model=model, temperature=temperature, max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    content = response.choices[0].message.content
    return json.loads(content), {"latency_seconds": round(time.monotonic() - started, 4), "usage": _response_usage(response)}


# =============================================================================
# 6. AGENT 1 — STRATEGY PLANNER
# =============================================================================

def validate_planner_decision(
    raw: Mapping[str, Any],
    shortlist: Sequence[Mapping[str, Any]],
    role: str,
    batna: int,
    round_index: int,
    state: Mapping[str, Any],
    must_resolve: bool,
    final_round: int,
) -> dict[str, Any]:
    action = str(raw.get("action", "")).strip().lower()
    action = {
        "counteroffer": "offer", "counter_offer": "offer", "propose": "offer", "make_offer": "offer",
        "reject": "walk_away", "impasse": "walk_away", "question": "ask", "clarify": "ask",
        "wait": "hold", "acknowledge": "hold", "discuss": "hold", "inform": "hold",
    }.get(action, action)
    if action not in ACTIONS:
        raise ValueError(f"invalid action: {action!r}")
    if action in TERMINAL_ACTIONS and round_index < final_round:
        raise ValueError(f"accept/walk_away are only available in round {final_round}")
    if must_resolve and action not in TERMINAL_ACTIONS:
        raise ValueError("the final scheduled turn requires accept or walk_away")

    focus_raw = raw.get("focus_issues") or []
    if not isinstance(focus_raw, list):
        raise ValueError("focus_issues must be an array")
    focus_issues = [str(issue).lower() for issue in focus_raw]
    if len(focus_issues) != len(set(focus_issues)) or any(issue not in ISSUES for issue in focus_issues):
        raise ValueError("focus_issues has an unknown or duplicate issue")

    reopen_raw = raw.get("reopen_issues") or []
    if not isinstance(reopen_raw, list):
        raise ValueError("reopen_issues must be an array")
    reopen_issues = [str(issue).lower() for issue in reopen_raw]
    if any(issue not in ISSUES for issue in reopen_issues):
        raise ValueError("reopen_issues has an unknown issue")

    read_of_opponent = str(raw.get("read_of_opponent", "") or "").strip()
    note = str(raw.get("note", "") or "").strip()
    candidate_id, scope, package_out, terms = None, "none", None, {}

    if action == "offer":
        candidate_id = str(raw.get("candidate_id", "") or "").strip()
        by_id = {c["candidate_id"]: c for c in shortlist}
        if candidate_id not in by_id:
            raise ValueError("offer requires a candidate_id copied from suggested_packages")
        candidate = by_id[candidate_id]["package"]
        scope = str(raw.get("scope", "full") or "full").lower()
        if scope not in ("full", "partial"):
            raise ValueError("scope must be 'full' or 'partial'")
        if scope == "partial":
            if not 1 <= len(focus_issues) <= 2:
                raise ValueError("a partial offer needs one or two focus_issues")
            terms = {issue: candidate[issue] for issue in focus_issues}
            package_out = None
        else:
            terms = dict(candidate)
            package_out = candidate
            if not focus_issues:
                own_position = state.get("current_position_by_role", {}).get(role, {})
                focus_issues = [i for i in ISSUES if own_position.get(i) != candidate[i]] or list(ISSUES)
        settled = state.get("settled_issues", {})
        contradicted = [i for i, v in terms.items() if i in settled and settled[i] != v and i not in reopen_issues]
        if contradicted:
            raise ValueError(
                f"{contradicted} already settled; list them in reopen_issues to reopen, or drop them from this offer"
            )
    elif action == "accept":
        outstanding = state.get("outstanding_offer")
        if not outstanding or outstanding.get("offered_by") == role:
            raise ValueError("nothing to accept: no pending offer from the opponent")
        package = outstanding.get("package")
        if package is None:
            raise ValueError("can only accept a complete package, not a single term")
        if score_package(role, package) <= batna:
            raise ValueError("cannot accept a package that does not beat BATNA")
        terms, scope, package_out, focus_issues = dict(package), "full", package, []
    elif action in ("ask", "hold"):
        if not 1 <= len(focus_issues) <= 2:
            raise ValueError(f"{action} needs one or two focus_issues")
    elif action == "greet":
        # Hard backstop: no matter how the planner misreads a line, it cannot pick greet
        # once real numbers are already on the table -- that judgment call only matters
        # in the pre-numeric window, where the model's own reading of the actual last
        # line (not a keyword/round heuristic) decides greet vs. a substantive action.
        has_numbers = any(bool(pos) for pos in state.get("current_position_by_role", {}).values())
        if has_numbers:
            raise ValueError("greet is only valid before any numbers have been exchanged")
        # Second backstop: this role already greeted (free-running's dedicated round-0
        # greeting round marks this -- see simulate_multi_agent_dyad). Without it, a
        # cautious planner can keep reading the still-no-numbers state as "still just
        # small talk" and pick greet again in round 1+, producing a second round of
        # bare "hi"s before any real content -- wastes a whole round out of 6. Always
        # False for replay, which never populates last_action_by_role.
        if state.get("last_action_by_role", {}).get(role) == "greet":
            raise ValueError("greet is only valid once -- you already greeted, move to substance now")
        focus_issues = []
    else:  # walk_away
        focus_issues = []

    return {
        "action": action, "candidate_id": candidate_id, "scope": scope, "package": package_out,
        "terms": terms, "focus_issues": focus_issues,
        "reopen_issues": reopen_issues if action == "offer" else [],
        "read_of_opponent": read_of_opponent, "note": note,
    }


def plan_action(
    client: Any, model: str, role: str, opponent: str, round_index: int, batna: int,
    turns: Sequence[Mapping[str, Any]], state: Mapping[str, Any], shortlist: Sequence[Mapping[str, Any]],
    must_resolve: bool, memory: Sequence[Mapping[str, Any]], final_round: int = 6, attempts: int = 3,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    positions = state.get("current_position_by_role", {})
    settled = state.get("settled_issues", {})
    open_issues = [issue for issue in ISSUES if issue not in settled]
    compact_shortlist = [{k: v for k, v in c.items() if k != "own_normalized"} for c in shortlist]

    system = (
        "You are the Strategy Planner for one side of a live compensation negotiation. Each turn, first read "
        "what the other side's last move actually signals, then decide what to do next. You privately know your "
        "own payoff table and walk-away point (BATNA); the other side never sees either, and you never reveal them. "
        "Issues in settled_issues are things both sides are already at the same number on -- once that happens, "
        "move the conversation to whichever issue in open_issues is still unresolved instead of re-litigating a "
        "settled one, the way a real negotiator naturally would. "
        "Your options: greet (a purely social reply with nothing about the negotiation -- appropriate ONLY when "
        "the other side's last line was itself pure small talk, like a hello or how-are-you, with nothing to "
        "respond to; the moment they ask or state anything relevant -- even something with no numbers in it, "
        "like 'what do you value most?' -- that is real content and calls for ask/offer/hold, not greet), offer "
        "(propose a package or just the term(s) you're focused on), accept (take the other "
        "side's pending complete offer), ask (a real question about their priorities or flexibility), hold "
        "(restate where you stand, or signal you're ready to move to the next open issue, without new numbers), "
        "and walk_away (end without a deal). "
        "suggested_packages is your offer calculator's shortlist: every candidate already beats your BATNA, respects "
        "settled_issues, and doesn't quietly take back ground you've already given, ranked by your own payoff and by "
        "how close it sits to what the other side has actually asked for. When you offer, copy one candidate_id from "
        "that list rather than inventing numbers. scope 'full' communicates the whole package; 'partial' names just "
        "the focus_issues you're proposing this turn -- normal, and often the more natural move than restating every "
        "term every round. "
        "accept and walk_away only become available once the negotiation reaches its final round, so keep negotiating "
        "even if a deal already looks acceptable -- the conversation is meant to run its full course. "
        "scope 'partial' is the DEFAULT for offers -- real negotiators rarely re-announce the whole package; use "
        "'full' only when you are genuinely proposing a complete deal for the first time or re-confirming one at "
        "the end. Likewise, not every turn needs new numbers: when the other side's last move doesn't change "
        "anything for you, a plain hold (or a short accept once the deal is clearly done) is the natural next "
        "move -- real negotiators spend a large share of turns on brief acknowledgments and logistics, not fresh "
        "proposals every time. "
        "Return JSON with: read_of_opponent (1-2 sentences on what their last move told you), action, candidate_id, "
        "scope, focus_issues, reopen_issues (only if deliberately reopening a settled issue), and "
        "note (a short private steer for whoever writes the actual chat message, e.g. 'push back gently before "
        "landing here' or 'we're aligned, signal moving on'). Do not write the public chat message yourself."
    )
    user = json.dumps({
        "role": role, "opponent": opponent, "round": round_index, "final_round": final_round,
        "rounds_remaining_including_current": final_round - round_index + 1,
        "must_resolve_now": must_resolve,
        "private_batna": batna,
        "payoff_table": UTILITY[canonical_role(role)],
        "level_values": LEVEL_VALUES,
        "settled_issues": settled,
        "open_issues": open_issues,
        "your_current_position": positions.get(role, {}),
        "opponent_current_position": positions.get(opponent, {}),
        "outstanding_offer": state.get("outstanding_offer"),
        "already_asked_about": state.get("asked_about_by_role", {}).get(role, []),
        "recent_conversation": [dict(t) for t in turns[-10:]],
        "suggested_packages": compact_shortlist,
        "recent_own_moves": list(memory[-4:]),
    }, ensure_ascii=False)

    traces: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        raw: Mapping[str, Any] | None = None
        try:
            raw, meta = call_json(client, model, system, user, temperature=0.35, max_tokens=500)
            decision = validate_planner_decision(
                raw, shortlist, role, batna, round_index, state, must_resolve, final_round,
            )
            traces.append({"attempt": attempt, "raw": raw, "validation_error": None, **meta})
            return decision, traces
        except (ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
            traces.append({"attempt": attempt, "raw": raw, "validation_error": str(exc)})
            user += f"\nPrevious output was invalid: {exc}. Return corrected JSON."

    fallback = _fallback_decision(role, batna, state, shortlist, must_resolve, open_issues)
    traces.append({"attempt": "deterministic_fallback", "raw": fallback, "validation_error": None})
    return fallback, traces


def _fallback_decision(
    role: str, batna: int, state: Mapping[str, Any], shortlist: Sequence[Mapping[str, Any]],
    must_resolve: bool, open_issues: Sequence[str],
) -> dict[str, Any]:
    if must_resolve:
        outstanding = state.get("outstanding_offer") or {}
        package = outstanding.get("package") if outstanding.get("offered_by") != role else None
        if package and score_package(role, package) > batna:
            return {
                "action": "accept", "candidate_id": None, "scope": "full", "package": package,
                "terms": dict(package), "focus_issues": [], "reopen_issues": [],
                "read_of_opponent": "", "note": "fallback: taking the pending offer",
            }
        return {
            "action": "walk_away", "candidate_id": None, "scope": "none", "package": None,
            "terms": {}, "focus_issues": [], "reopen_issues": [],
            "read_of_opponent": "", "note": "fallback: no viable deal on the table",
        }
    if shortlist:
        # Default to a PARTIAL offer, matching plan_action's own steer that partial is the
        # normal move -- a fallback should not be the one path that always produces a
        # full-package (and therefore always Integrative-coded) turn.
        top = shortlist[0]
        focus = (list(open_issues) or list(ISSUES))[:2]
        return {
            "action": "offer", "candidate_id": top["candidate_id"], "scope": "partial", "package": None,
            "terms": {issue: top["package"][issue] for issue in focus}, "focus_issues": focus,
            "reopen_issues": [], "read_of_opponent": "",
            "note": "fallback: proposing the open issue(s) from the top-ranked package",
        }
    focus = [open_issues[0]] if open_issues else [ISSUES[0]]
    return {
        "action": "hold", "candidate_id": None, "scope": "none", "package": None, "terms": {},
        "focus_issues": focus, "reopen_issues": [], "read_of_opponent": "",
        "note": "fallback: nothing left to concede on, holding position",
    }


@dataclass
class StrategyPlannerAgent:
    client: Any
    model: str
    role: str
    opponent: str
    memory: list[dict[str, Any]]

    @classmethod
    def create(cls, client: Any, model: str, role: str, opponent: str) -> "StrategyPlannerAgent":
        return cls(client=client, model=model, role=role, opponent=opponent, memory=[])

    def act(
        self, round_index: int, batna: int, turns: Sequence[Mapping[str, Any]], state: Mapping[str, Any],
        shortlist: Sequence[Mapping[str, Any]], must_resolve: bool, final_round: int = 6, attempts: int = 3,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        decision, traces = plan_action(
            self.client, self.model, self.role, self.opponent, round_index, batna, turns, state,
            shortlist, must_resolve, self.memory, final_round=final_round, attempts=attempts,
        )
        self.memory.append({
            "round": round_index, "action": decision["action"], "scope": decision["scope"],
            "terms": decision["terms"], "focus_issues": decision["focus_issues"],
        })
        self.memory[:] = self.memory[-6:]
        return decision, traces


# =============================================================================
# 7. AGENT 2 — LANGUAGE REALIZER
# =============================================================================

_BARE_NUMBER_LIST = re.compile(r"^[\d,.\s$%]+$")

# A short acknowledgment/filler fused onto the front of a message that also states real
# terms -- e.g. "understood, but 120k salary and 20k stock options are where i need to
# be" -- reads as a form filling in a template, not a person typing two thoughts. Real
# chat sends these as separate messages: "understood." / "120k salary and 20k stock
# options are where i need to be."
_ACK_PREFIX = re.compile(
    r"^(ok(ay)?|understood|got it|sounds good|noted|agreed|fair enough|makes sense|"
    r"i see|good to know|appreciate (that|it))[,.]?\s+(but|so|and)?\s*\w",
    re.IGNORECASE,
)
# Same problem, mirrored: real content followed by a short closing tag in the same
# message -- e.g. "120k salary, 20% bonus, 20k stock, 10 days vacation. that's my
# best." Splitting isolates the closing remark as its own line, the way a person
# would actually send it.
_CLOSING_SUFFIX = re.compile(
    r"\.\s*(that'?s (my|the) (best|final|top)(\s+offer)?|that'?s it|final answer|"
    r"take it or leave it)\.?\s*$",
    re.IGNORECASE,
)


def validate_realization(messages: Any, decision: Mapping[str, Any]) -> list[str]:
    if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_BURST_MESSAGES:
        return [f"messages must be a list of 1 to {MAX_BURST_MESSAGES} strings"]
    if any(not isinstance(m, str) or not m.strip() for m in messages):
        return ["each message must be a nonempty string"]
    errors: list[str] = []
    if any(len(m.split()) > 20 for m in messages):
        errors.append("keep each message to at most 20 words")
    if any(_BARE_NUMBER_LIST.match(m.strip()) for m in messages):
        errors.append(
            "a message must not be a bare list of numbers separated by commas -- this is a chat message, "
            "not a data dump; use words (salary, bonus, %, k, stock, days) or a short natural phrase"
        )
    fused = [m for m in messages
             if parse_offer(m) and (_ACK_PREFIX.match(m.strip()) or _CLOSING_SUFFIX.search(m.strip()))]
    if fused:
        errors.append(
            f"{fused!r} fuses a short acknowledgment or closing remark onto real terms in one message -- "
            "split them into two separate messages: the acknowledgment/closing tag as its own short line, "
            "the terms as a separate one, e.g. 'understood.' + 'that works for salary, what about bonus?' "
            "not 'understood, that works for salary...' -- real people send these as two texts. "
            "The above is illustrative only -- do not reuse 'understood' or any other wording from these "
            "examples verbatim. Write the acknowledgment and the offer in your own words, varying phrasing "
            "across turns so it doesn't read as a template."
        )
    # Hard brevity floor for the two actions that introduce nothing new: without this, the
    # realizer never takes the terse branch it's permitted to take, and every accept/hold
    # turn gets padded into something warm enough to be classified Integrative instead of
    # Neutral. accept always carries the accepted package as terms (that's what was agreed,
    # not new content this speaker is introducing), so it is checked unconditionally; hold
    # only qualifies when it truly has no new terms (real per plan_action/validate_planner_decision).
    action = decision.get("action")
    total_words = sum(len(m.split()) for m in messages)
    if action == "accept" and total_words > 5:
        errors.append(
            "action='accept' has nothing new to say -- you're just taking the other side's offer, so the "
            "reply should be a bare fragment (e.g. 'ok', 'deal', 'that works') of at most 5 words total, "
            "not an explanation"
        )
    if action == "hold" and not decision.get("terms") and total_words > 8:
        errors.append(
            "action='hold' introduces no new numbers -- keep the reply to at most 8 words: a short "
            "restatement or a nudge to the next issue, not an explanation"
        )
    parsed = parse_messages(messages)
    intended = dict(decision.get("package") or decision.get("terms") or {})
    contradiction = {i: v for i, v in parsed.items() if i in intended and intended[i] != v}
    if contradiction:
        errors.append(f"message states {contradiction}, which contradicts the decision's terms {intended}")
    return errors


def deterministic_realization(decision: Mapping[str, Any]) -> list[str]:
    action = decision["action"]
    if action == "offer":
        if decision.get("scope") == "full" and decision.get("package"):
            content = [f"I could do {format_package(decision['package'])}."]
        else:
            terms = decision.get("terms") or {}
            content = [f"What about {_format_terms(terms)}?"] if terms else ["What would work for you here?"]
    elif action == "accept":
        return ["That works for me."]
    elif action == "walk_away":
        return ["I don't think we're going to land on something here."]
    elif action == "ask":
        focus = decision.get("focus_issues") or []
        content = [f"What matters more to you, {focus[0]}?"] if focus else ["What matters most to you here?"]
    else:
        focus = decision.get("focus_issues") or []
        return [f"Let's talk about {focus[0]} next."] if focus else ["I'll hold here for now."]
    return content


def realize_action(
    client: Any, model: str, role: str, decision: Mapping[str, Any], turns: Sequence[Mapping[str, Any]],
    examples: Sequence[str], attempts: int = 4,
) -> tuple[list[str], list[dict[str, Any]]]:
    history = "\n".join(f"{t['role']}: {t['text']}" for t in turns[-8:]) or "(you speak first)"
    last_line = str(turns[-1]["text"]) if turns else None
    system = (
        "Write this negotiator's decision as natural chat and return JSON as {\"messages\": [...]}. This is a "
        "conversation, not a form submission: read opponent_last_line and write like you're actually replying to "
        "it, not reciting your position in a vacuum. Never send a bare list of numbers with nothing else (e.g. "
        "'120000, 20, 20000, 30') -- that's a data dump, not chat; say the numbers the way a person would, with "
        "words attached (salary, bonus, %, k, stock, days), or wrap them in a short phrase or question. "
        "Real negotiators are still terse, though: a bare number with its unit, a short fragment ('ok', 'deal', "
        "'that works'), or one short question are all complete messages on their own -- don't pad them into full "
        "sentences. When the decision is accept, or hold with nothing new in its terms, a short fragment IS the "
        "whole message: do not add unprompted appreciation, warmth, or 'let's find something that works for "
        "both of us' framing just to fill space -- save that language for when the decision's note actually "
        "calls for relationship repair. Most turns are business-like, not warm. Aim for roughly 6-12 words when "
        "you do write a full sentence. One message is the default, but you can send up to a few short ones in a "
        "row when they genuinely do different jobs -- reacting to what they just said, then asking or proposing "
        "something new, is a normal chat rhythm; don't fuse a reaction and new content into one run-on line just "
        "because one message is the default -- e.g. 'ok, that works' then, separately, 'what about the vacation "
        "days?' reads like a real person, 'ok that works, what about the vacation days' reads like a form. The "
        "same applies at the very start: if this is your first substantive message and the conversation so far "
        "has been purely social (a hello, nothing about the negotiation from either side yet), a brief greeting "
        "line or two is a natural lead-in before the real content -- e.g. 'hi!' then, separately, 'thanks for "
        "having me' then, separately, your actual offer -- rather than skipping straight to numbers with zero "
        "transition. Don't force this once the conversation is already underway; it's for the opening only. "
        "This rule applies throughout, not just reactions and openings: whenever a short acknowledgment or "
        "closing remark and real terms end up in the same sentence, split them. Wrong: '120k salary, 20% bonus, 20k stock, 10 "
        "days vacation. that's my best.' -- right: '120k salary, 20% bonus, 20k stock, 10 days vacation.' "
        "then, separately, 'that's my best.' A comma or a period does not count as separating two thoughts -- "
        "only two actual messages do. Acknowledging is often a good move, but it is not required every time -- "
        "skip it when it would feel redundant or slow the exchange down. "
        "Casual is normal: lowercase is fine, contractions are natural (I'm, that's, "
        "can't). State only the numbers present in the decision's terms -- don't restate settled issues and "
        "don't invent new ones. This is a live, self-contained negotiation: never mention BATNA, payoff, "
        "strategy, or defer to an outside approval, manager, or later meeting."
    )
    user = json.dumps({
        "role": role, "decision": decision, "opponent_last_line": last_line, "conversation": history,
    }, ensure_ascii=False) + format_human_style_examples(examples)

    traces: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        raw: Mapping[str, Any] | None = None
        try:
            raw, meta = call_json(client, model, system, user, temperature=1.0, max_tokens=70)
            messages = raw.get("messages")
            errors = validate_realization(messages, decision)
            traces.append({"attempt": attempt, "raw": raw, "validation_errors": errors, **meta})
            if not errors:
                return [str(m).strip() for m in messages], traces
            user += f"\nPrevious output errors: {'; '.join(errors)}. Return a fresh valid JSON messages array."
        except (ValueError, TypeError, AttributeError, json.JSONDecodeError) as exc:
            traces.append({"attempt": attempt, "raw": raw, "validation_errors": [str(exc)]})

    fallback = deterministic_realization(decision)
    traces.append({"attempt": "deterministic_fallback", "raw": {"messages": fallback}, "validation_errors": []})
    return fallback, traces


@dataclass
class LanguageRealizerAgent:
    client: Any
    model: str
    role: str
    memory: list[dict[str, Any]]

    @classmethod
    def create(cls, client: Any, model: str, role: str) -> "LanguageRealizerAgent":
        return cls(client=client, model=model, role=role, memory=[])

    def act(
        self, decision: Mapping[str, Any], turns: Sequence[Mapping[str, Any]], examples: Sequence[str],
        attempts: int = 4,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        messages, traces = realize_action(self.client, self.model, self.role, decision, turns, examples, attempts)
        self.memory.append({"messages": messages})
        self.memory[:] = self.memory[-6:]
        return messages, traces


# =============================================================================
# 7b. ROUND 0 — A SHORT OPENING GREETING, BEFORE ANY NEGOTIATION CONTENT
# =============================================================================
# Real dyads don't open with an offer -- round 0 in the human corpus is almost
# entirely "hello" / "hi there, nice to meet you" style Neutral small talk.
# This doesn't touch private payoff/BATNA and doesn't go through the offer
# calculator or the planner at all; it's just the realizer writing one short
# line, gated by a check that no negotiation numbers snuck in.

GREETING_FALLBACKS = ("Hi, nice to meet you.", "Hello!", "Hey, ready when you are.")


def validate_greeting(messages: Any) -> list[str]:
    if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_GREETING_MESSAGES:
        return [f"a greeting is 1 to {MAX_GREETING_MESSAGES} short messages"]
    if any(not isinstance(m, str) or not m.strip() for m in messages):
        return ["each message must be a nonempty string"]
    errors = []
    if any(len(m.split()) > 12 for m in messages):
        errors.append("keep each message to at most 12 words")
    if any(parse_offer(m) for m in messages):
        errors.append("a greeting must not mention any negotiation numbers yet")
    return errors


def realize_greeting(
    client: Any, model: str, role: str, turns: Sequence[Mapping[str, Any]],
    examples: Sequence[str], attempts: int = 3,
) -> tuple[list[str], list[dict[str, Any]]]:
    history = "\n".join(f"{t['role']}: {t['text']}" for t in turns[-4:]) or "(you speak first)"
    system = (
        "Write this negotiator's opening small talk, before the negotiation itself starts -- just real-chat "
        "hellos, nothing about salary, bonus, stock, or vacation yet. Return JSON as "
        "{\"messages\": [\"...\"]}, 1 to 3 short messages, each at most 12 words. One message is the most "
        "common case, but real people sometimes send a couple of quick lines in a row before the other side "
        "replies -- e.g. 'hi!' then, separately, 'nice to meet you' -- so use two or three only when it "
        "genuinely reads that way, not as a rule to follow every time. Casual is normal: lowercase is fine, "
        "a bare 'hey' or 'hi there' is a complete message on its own. Match the tone of the human examples "
        "below without copying their exact wording."
    )
    user = json.dumps({"role": role, "conversation": history}, ensure_ascii=False) + format_human_style_examples(examples)

    traces: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        raw: Mapping[str, Any] | None = None
        try:
            raw, meta = call_json(client, model, system, user, temperature=1.0, max_tokens=90)
            messages = raw.get("messages")
            errors = validate_greeting(messages)
            traces.append({"attempt": attempt, "raw": raw, "validation_errors": errors, **meta})
            if not errors:
                return [str(m).strip() for m in messages], traces
            user += f"\nPrevious output errors: {'; '.join(errors)}. Return a fresh valid JSON messages array."
        except (ValueError, TypeError, AttributeError, json.JSONDecodeError) as exc:
            traces.append({"attempt": attempt, "raw": raw, "validation_errors": [str(exc)]})

    fallback = [GREETING_FALLBACKS[hash((role, len(turns))) % len(GREETING_FALLBACKS)]]
    traces.append({"attempt": "deterministic_fallback", "raw": {"messages": fallback}, "validation_errors": []})
    return fallback, traces


@dataclass
class TwoAgentTeam:
    planner: StrategyPlannerAgent
    realizer: LanguageRealizerAgent

    @classmethod
    def create(cls, client: Any, model: str, role: str, opponent: str) -> "TwoAgentTeam":
        return cls(
            planner=StrategyPlannerAgent.create(client, model, role, opponent),
            realizer=LanguageRealizerAgent.create(client, model, role),
        )


# =============================================================================
# 8. ORCHESTRATION
# =============================================================================

@dataclass
class MultiAgentConfig:
    model: str = "gpt-5.4"
    initial_batna: int = 35
    new_batna: int = 50
    batna_round: int = 4
    shortlist_size: int = 12
    final_round: int = 6
    greeting_round: bool = True


def speaking_order(rng: Any, roles: Sequence[str]) -> list[str]:
    """Who goes first in this dyad -- shared by the round-0 greeting and the schedule below."""
    if len(roles) != 2:
        raise ValueError("exactly two roles are required")
    ordered = list(roles)
    if float(rng.random()) < 0.5:
        ordered.reverse()
    return ordered


def alternating_schedule(ordered_roles: Sequence[str], rounds: int) -> list[tuple[int, str]]:
    """One turn per role per round, in the given speaking order."""
    return [(round_index, role) for round_index in range(1, rounds + 1) for role in ordered_roles]


def burst_metrics(turns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    burst_sizes: dict[str, int] = {}
    for turn in turns:
        burst_sizes[str(turn["strategic_turn_id"])] = int(turn["burst_size"])
    utterances, bursts = len(turns), len(burst_sizes)
    return {
        "strategic_turns": bursts,
        "utterances": utterances,
        "utterances_per_strategic_turn": round(utterances / bursts, 4) if bursts else 0,
        "multi_message_burst_rate": round(sum(s > 1 for s in burst_sizes.values()) / bursts, 4) if bursts else 0,
        "mean_words_per_utterance": round(
            sum(len(str(t["text"]).split()) for t in turns) / utterances, 4
        ) if utterances else 0,
    }


ACTION_TO_BEHAVIOR = {
    "offer": "Integrative", "accept": "Neutral", "ask": "Integrative",
    "hold": "Distributive", "walk_away": "Distributive", "greet": "Neutral",
}


def simulate_multi_agent_dyad(
    client: Any,
    game_id: str,
    roles: Sequence[str],
    rng: Any,
    batna_target: str,
    retrieve_examples: Callable[[str, int, str, str | None, str], tuple[list[str], list[Any]]],
    config: MultiAgentConfig | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = config or MultiAgentConfig()
    turns: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    batnas = {role: config.initial_batna for role in roles}
    batna_delivered = {role: False for role in roles}
    state = initial_shared_state(roles)
    ordered = speaking_order(rng, roles)
    schedule = alternating_schedule(ordered, rounds=config.final_round)
    teams = {
        role: TwoAgentTeam.create(client, config.model, role, next(o for o in roles if o != role))
        for role in roles
    }
    strategic_turn_number = 0

    if config.greeting_round:
        for role in ordered:
            strategic_turn_number += 1
            history = "\n".join(f"{t['role']}: {t['text']}" for t in turns[-4:])
            examples, example_ids = retrieve_examples(role, 0, "Pre-BATNA", "Neutral", history)
            messages, greet_attempts = realize_greeting(client, config.model, role, turns, examples)

            strategic_turn_id = f"{game_id}_turn_{strategic_turn_number:02d}"
            for message_index, message in enumerate(messages, start=1):
                turns.append({
                    "actor_id": f"{game_id}_{role}", "role": role, "round": 0, "phase": "Pre-BATNA",
                    "text": message, "strategic_turn_id": strategic_turn_id,
                    "message_index": message_index, "burst_size": len(messages),
                })

            state_before = json.loads(json.dumps(state))
            # Mark this role as having greeted, so a later planner call in the regular
            # schedule (round 1+) can't pick "greet" again just because no numbers have
            # been exchanged yet -- validate_planner_decision's greet guard checks this.
            # Reuses last_action_by_role rather than adding new state: replay's context
            # reconstruction never populates that field (always None there), so this is
            # a no-op for replay -- scoped to free-running's dedicated greeting round only.
            state["last_action_by_role"][role] = "greet"
            state_after = json.loads(json.dumps(state))
            traces.append({
                "game_id": game_id, "strategic_turn_id": strategic_turn_id, "round": 0, "role": role,
                "batna": batnas[role], "batna_update_received": False,
                "shortlist": [], "planner_decision": {
                    "action": "greet", "candidate_id": None, "scope": "none", "package": None,
                    "terms": {}, "focus_issues": [], "reopen_issues": [],
                    "read_of_opponent": "", "note": "opening greeting, before negotiation content",
                },
                "planner_attempts": [], "realizer_attempts": greet_attempts,
                "state_before": state_before, "state_after": state_after,
                "retrieval_ids": list(example_ids), "retrieved_examples": list(examples),
                "fallback_used": any(a.get("attempt") == "deterministic_fallback" for a in greet_attempts),
                "burst_size": len(messages),
            })

    for turn_number, (round_index, role) in enumerate(schedule, start=1):
        opponent = next(o for o in roles if o != role)
        received_update = (
            role == batna_target and round_index == config.batna_round and not batna_delivered[role]
        )
        if received_update:
            batnas[role] = config.new_batna
            batna_delivered[role] = True
            state["batna_updates"][role] = True

        own_position = state["current_position_by_role"].get(role, {})
        opponent_position = state["current_position_by_role"].get(opponent, {})
        shortlist = rank_offers(
            role, batnas[role], own_position, opponent_position, state.get("settled_issues", {}),
            limit=config.shortlist_size,
        )
        must_resolve = turn_number == len(schedule)
        decision, planner_attempts = teams[role].planner.act(
            round_index, batnas[role], turns, state, shortlist, must_resolve, final_round=config.final_round,
        )

        phase = "Pre-BATNA" if round_index < config.batna_round else "Post-BATNA"
        history = "\n".join(f"{t['role']}: {t['text']}" for t in turns[-8:])
        # The planner has already decided this turn's action, so retrieval can target
        # exactly the behavior category this message will land in -- no need for the
        # single-agent generators' "last turn's category as a proxy" workaround, and no
        # gap on turn one where no prior category exists yet.
        current_behavior = ACTION_TO_BEHAVIOR[decision["action"]]
        examples, example_ids = retrieve_examples(role, round_index, phase, current_behavior, history)
        messages, realizer_attempts = teams[role].realizer.act(decision, turns, examples)

        strategic_turn_number += 1
        strategic_turn_id = f"{game_id}_turn_{strategic_turn_number:02d}"
        for message_index, message in enumerate(messages, start=1):
            turns.append({
                "actor_id": f"{game_id}_{role}", "role": role, "round": round_index, "phase": phase,
                "text": message, "strategic_turn_id": strategic_turn_id,
                "message_index": message_index, "burst_size": len(messages),
            })

        state_before = json.loads(json.dumps(state))
        state = update_shared_state(state, role, decision, messages)

        traces.append({
            "game_id": game_id, "strategic_turn_id": strategic_turn_id, "round": round_index, "role": role,
            "batna": batnas[role], "batna_update_received": received_update,
            "shortlist": shortlist, "planner_decision": decision, "planner_attempts": planner_attempts,
            "realizer_attempts": realizer_attempts,
            "state_before": state_before, "state_after": state,
            "retrieval_ids": list(example_ids), "retrieved_examples": list(examples),
            "fallback_used": any(a.get("attempt") == "deterministic_fallback" for a in realizer_attempts),
            "burst_size": len(messages),
        })

        if decision["action"] in TERMINAL_ACTIONS and round_index >= config.final_round:
            break

    terminal = traces[-1]["planner_decision"]["action"] if traces else None
    full_schedule = ([(0, role) for role in ordered] if config.greeting_round else []) + schedule
    public = {
        "game_id": game_id, "two_agent_version": TWO_AGENT_VERSION,
        "turns": turns, "batna_target": batna_target,
        "schedule": [{"round": r, "role": role} for r, role in full_schedule],
        "message_metrics": burst_metrics(turns),
        "deal": state.get("deal"),
        "settled_issues": state.get("settled_issues", {}),
        "terminal_action": terminal if terminal in TERMINAL_ACTIONS else None,
    }
    return public, traces
