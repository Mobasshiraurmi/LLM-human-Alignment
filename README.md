# LLM negotiator alignment — paper code release

Code and data for comparing how closely LLM-driven negotiators match real human negotiators,
across three agent architectures, three backend models, and two experimental designs.

- **Architectures**: `single` (one direct system-prompted call per turn), `multi` (a
  strategy-planner + language-realizer pair, `two_agents.py`), `single_personality` (the
  single-agent architecture with a sampled Big-Five personality per role, `personality.py`).
- **Models**: GPT-5.4, Claude Sonnet 4.6, Gemini 2.5 Pro — swapped via `llm_backends.py`, which
  gives every provider the same `client.chat.completions.create(...)` surface so the prompts,
  validation and retries are identical across models.
- **Designs**: `replay` (the model writes one reply at each real human turn, using only the
  real prior context each time — `step5_replay.py`) and `freerun` (the model plays a full
  6-round dialogue against itself from scratch — `step2_generate.py`).

That's 3 × 3 × 2 = 18 combinations, all already generated, classified (by a GPT-5.4 behavior
classifier, same coder for every model so classifier choice is never a confound) and checked
into `data/replay/<key>/` / `data/freerun/<key>/` (`<key>` is the architecture name, or
`<architecture>_<model>` for anything other than GPT-5.4).

## Quickstart

```bash
pip install -r requirements.txt
python3 run_pipeline.py
```

This rebuilds every report from the results already in the repo — no API key, no cost, done in
under a minute. It produces:

- **`report/kodis_report_{models,types_single,types_multi,architectures}.html`** (+ `.tex`) — the
  primary comparison tables: model-wise (GPT-5.4 vs Claude vs Gemini, per architecture),
  type-wise (replay vs free-running, for the single-agent baseline and separately for
  multi-agent), and architecture-wise (single vs multi vs single+personality, on GPT-5.4). Built
  by `kodis_reports.py` using `kodis_stats.py`'s divergence-gap engine — a corrected version of
  the KODIS paper's own evaluation procedure (see `kodis_stats.py`'s docstring: the published
  procedure's significance test has a measured 43–50% false-positive rate on this data; the
  permutation-of-dyads test used here brings that to 3.5–6.5%).
- **`data/replay/<key>/report.html`** / **`data/freerun/<key>/report.html`** — one secondary, per-run HTML
  report per combo (18 total): a single-row `kodis_stats.py` divergence table (same statistics
  as the 4 comparison reports above, just one architecture at a time) plus the full conversation
  log (human turn beside model turn, for replay).

## Regenerating from scratch

```bash
export OPENAI_API_KEY=...        # always required (generation on gpt-5.4, classification always)
export ANTHROPIC_API_KEY=...     # for claude-sonnet-4.6 combos
export GEMINI_API_KEY=...        # for gemini-2.5-pro combos

python3 run_pipeline.py --full                                   # all 18 combos
python3 run_pipeline.py --full --combo single claude-sonnet-4.6 replay   # just one
```

`--full` chains, per combo: `step5_replay.py`/`step2_generate.py --run` (generate) →
`classify_replay.py`/`classify_freerun.py --submit/--check/--parse` (classify via the Batch
API, falling back to `--sync` real-time classification if the batch stalls) → the report step
above. Every underlying script tracks what it's already written and skips past it, so an
interrupted `--full` run — including this script getting killed — is safe to just re-run.

Output lands in **`full_run/<design>/<key>/`, not `data/`** — a fresh `--full` run is kept
completely separate from the shipped, already-reviewed results, so there's no risk of
silently overwriting them. Compare your run against the shipped one, or pass `--out-dir data`
if you deliberately want to regenerate the shipped results in place.

This is slow and costs real money (full 6-round/replay generation plus classification across
18 combos); the default report-only mode above is what most people want. For a cheap sanity
check instead, add `--smoke --combo <architecture> <model> <design>` (a handful of dyads,
one combo).

## Layout

```
run_pipeline.py                      the one entry point (see Quickstart above)

data/
  human_data.csv                  the real human negotiation transcripts
  scores.csv                      per-player BATNA values (who got the outside-option change)
  personality_adjective.json      Big-Five trait -> adjective list, used by personality.py
  replay/<key>/, freerun/<key>/   generated data + results + reports, one dir per combo

generation/
  step1_calibration.py    human-only power/calibration check (no API key needed)
  step2_generate.py       free-running generation (single / multi / single_personality × model)
  step5_replay.py         replay generation (same architectures/models)
  two_agents.py           the multi-agent planner + realizer
  personality.py          Big-Five trait sampling for single_personality
  llm_backends.py         provider abstraction: gpt-5.4 / claude-sonnet-4.6 / gemini-2.5-pro

classify/
  classify_replay.py      GPT-5.4 behavior classifier for replay output
  classify_freerun.py     same, for free-running output
  classifier_bridge.py    shared low-level classifier plumbing

stats/
  rag_align.py            shared alignment-metric primitives (behavior mix, phase profile, …)
  kodis_stats.py          the corrected KODIS-style divergence-gap engine (statistics)
  step6_paired.py         replay-only paired analysis (TOST equivalence, Gwet's AC1, …)

report/
  kodis_reports.py        registry-driven report builder on top of stats/kodis_stats.py
  report_registry.json    the 4 reports' row definitions (which results.csv goes where)
  report_replay.py        per-run HTML report, replay (uses kodis_stats.py for its table)
  report_freerun.py       per-run HTML report, free-running (same)
```

All scripts are run from the repo root (`python3 generation/step2_generate.py ...`, etc.) --
`run_pipeline.py` already does this for you.
