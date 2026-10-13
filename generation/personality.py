"""personality.py — Big-Five personality trait sampling for negotiation agents.

Ported from Kwon et al., "Evaluating Behavioral Alignment in Conflict Dialogue"
(EMNLP 2025), see:
  Eval-LLM-BehavAlign-main/src/simulation/personality.py
  Eval-LLM-BehavAlign-main/prompts/final/personality_adjective.json

The empirical priors and adjective lists are unchanged from that paper. Only
the interface is adapted: this repo's dyads are HR/Employee (not buyer/
seller), and personality is sampled ONCE per role per dyad (a person's
personality does not change mid-negotiation), with an explicit rng so
generation stays reproducible across resumed runs.
"""
import json
import os
import random
from itertools import chain
from typing import Dict, List, Optional, Tuple

# Personality priors based on empirical data (Kwon et al., EMNLP 2025).
# Each list is 6 weights over POLARITY_DEGREE_PAIRS: [Low-low, Low-medium,
# Low-high, High-low, High-medium, High-high].
PERSONALITY_POLARITY_DEGREE_PRIORS = {
    "EXT": [0.35, 0.35, 0.52, 0.52, 0.13, 0.13],
    "AGR": [0.09, 0.09, 0.38, 0.38, 0.54, 0.54],
    "CON": [0.05, 0.05, 0.39, 0.39, 0.55, 0.55],
    "NEU": [0.31, 0.31, 0.44, 0.44, 0.24, 0.24],
    "OPE": [0.04, 0.04, 0.49, 0.49, 0.47, 0.47],
}

INTENSITY_MAPPING = {"high": "very ", "medium": "", "low": "a bit "}

POLARITY_DEGREE_PAIRS = [
    "Low-low", "Low-medium", "Low-high",
    "High-low", "High-medium", "High-high",
]

DEFAULT_ADJECTIVE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "personality_adjective.json")


def load_personality_adjectives(path: str = DEFAULT_ADJECTIVE_PATH) -> Dict[str, Dict[str, List[str]]]:
    """Load the trait -> polarity -> adjective-list mapping (EXT/AGR/CON/NEU/OPE x Low/High)."""
    with open(path) as f:
        return json.load(f)


def select_adjectives(
    personality_dict: Dict[str, Dict[str, List[str]]],
    n: int = 3,
    intensity: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[Dict[str, Dict[str, List[str]]], str]:
    """Sample a Big-Five personality profile and render it as a sentence.

    For each trait (EXT/AGR/CON/NEU/OPE), draw a (polarity, degree) pair from
    the empirical prior, then sample n adjectives of that polarity, prefixed
    by the degree's intensity marker ("very "/""/"a bit ").

    Args:
        personality_dict: trait -> {"Low": [...], "High": [...]} adjective lists
        n: adjectives to sample per trait
        intensity: force 'high'/'medium'/'low' instead of the sampled degree
        rng: a random.Random instance for reproducible sampling; defaults to
            the module-level `random` (non-reproducible) if omitted

    Returns:
        (selected_adjectives, personality_sentence) — sentence is a ready-to-
        embed "You are X, Y, Z, ..." description of the sampled profile.
    """
    if intensity and intensity not in INTENSITY_MAPPING:
        raise ValueError("Intensity must be 'high', 'medium', or 'low'")
    rng = rng or random

    selected_adjectives = {}
    for trait, adjectives in personality_dict.items():
        prior = PERSONALITY_POLARITY_DEGREE_PRIORS[trait]

        polarity_degree = rng.choices(POLARITY_DEGREE_PAIRS, weights=prior)[0]
        polarity, degree = polarity_degree.split("-")
        degree = intensity or degree
        prefix = INTENSITY_MAPPING[degree]

        pool = adjectives[polarity]
        selected_adjectives[trait] = {
            polarity: [prefix + adj for adj in rng.sample(pool, min(n, len(pool)))]
        }

    combined = [
        adj
        for polarity_dict in selected_adjectives.values()
        for adj in polarity_dict.values()
    ]
    sentence = "You are " + ", ".join(chain(*combined)) + "."

    return selected_adjectives, sentence
