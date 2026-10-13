# LLM negotiator alignment — paper code release

Code and data for comparing how closely LLM-driven negotiators match real human negotiators,
across four agent architectures, three backend models, and two experimental designs.

- **Architectures**:
  - `single`: one direct system-prompted call per turn.
  - `single_personality`: the single agent with a sampled Big-Five personality per role
    (`personality.py`).
  - `single_rag`: the turn-grouped single agent with retrieved human turns as examples
    (`single_agent.py`, retrieval in `rag_align.py`).
  - `multi`: a strategy-planner + language-realizer pair (`two_agents.py`).
- **Models**: GPT-5.4, Claude Sonnet 4.6 and Gemini 2.5 Pro, swapped via `llm_backends.py`.
  It gives every provider the same `client.chat.completions.create(...)` surface, so the
  prompts, validation and retries are identical across models.
- **Designs**:
  - `replay`: the model writes one reply at each real human turn, using only the real prior
    context each time (`step5_replay.py`).
  - `freerun`: the model plays a full 6-round dialogue against itself from scratch
    (`step2_generate.py`).

That's 4 × 3 × 2 = 24 combinations, all already generated and classified. One GPT-5.4
behavior classifier codes every model's output, so the choice of classifier is never a
confound. The results are in `data/replay/<key>/` and `data/freerun/<key>/`, where `<key>` is
the architecture name, or `<architecture>_<model>` for any model other than GPT-5.4.

Every comparison uses the same human baseline: the 47 real dyads in `data/human_data.csv` with
a delivered BATNA change.

## Quickstart

```bash
pip install -r requirements.txt
python3 run_pipeline.py
```

This builds every report from the results already in the repo. It needs no API key, costs
nothing and takes about a minute. It writes everything to **`output/`** and changes no file
in the repo (`output/` is git-ignored). The reports are built by `report/alignment_reports.py`
and `report/meister_report.py` from the results listed in `report/results_catalog.json`
(all 24 combinations).

| File in `output/` | What it shows |
|---|---|
| `kwon_by_model.html`, `kwon_by_type.html`, `kwon_by_architecture.html` | Kwon-style **mean gaps** (agent mean − human mean), one combined table per comparison: ΔDistributive and the Pre/Post-BATNA Neutral / Integrative / Distributive shares. Paired t-tests for replay, Welch t-tests for free run. Statistics in `stats/kodis_stats.py`. |
| `gao_by_model.html`, `gao_by_type.html`, `gao_by_architecture.html` | Gao-style **distribution-level** divergence: Jensen–Shannon divergence between the agent's and the humans' distributions of each per-dyad value, with dyad-permutation tests (`stats/gao_style_table.py`). The by-architecture page adds the Gao et al. Fig. 4 panels. |
| `meister_replay_vs_freerun.html` | **Replay vs free run** for each architecture × model (Meister et al. 2025, Eq. 1), with paired-bootstrap CIs, Wilcoxon tests and a forest plot. |
| `figures/` | Every figure as a vector PDF and an SVG, drawn at its printed ACL size: one column (219 pt) or two columns (455 pt). |

- **By model** compares GPT-5.4, Claude and Gemini within each architecture and design.
- **By type** compares replay with free run within each architecture and model.
- **By architecture** compares the four architectures within each design and model.

Every `.html` page has a `.tex` twin with the same tables as LaTeX. The `.tex` files also
contain ready `figure` environments for the PDFs in `output/figures/` (`report/paper_figures.py`).
Exporting the PDFs needs Google Chrome; without it, only the SVGs are written.

To view the reports:

```bash
open output/*.html                       # macOS; or open any file in output/ in a browser
```

## Regenerating from scratch

```bash
export OPENAI_API_KEY=...        # always required (generation on gpt-5.4, classification always)
export ANTHROPIC_API_KEY=...     # for claude-sonnet-4.6 combos
export GEMINI_API_KEY=...        # for gemini-2.5-pro combos

python3 run_pipeline.py --full                                         # the 18 single / single_personality / multi combos
python3 run_pipeline.py --full --combo single claude-sonnet-4.6 replay # just one
python3 run_pipeline.py --full --combo single_rag gpt-5.4 freerun      # single_rag (only via --combo)
python3 run_pipeline.py --full --smoke --combo single gpt-5.4 freerun  # cheap check: a handful of dyads
```

For each combination, `--full` runs three steps:
1. **Generate** with `step5_replay.py` or `step2_generate.py --run`.
2. **Classify** with `classify_replay.py` or `classify_freerun.py --submit/--check/--parse`,
   through the OpenAI Batch API. If the batch stalls, it falls back to `--sync` real-time
   classification.
3. **Report**: the reports are rebuilt at the end, but only when the run wrote into `data/`,
   the folder the catalog points to.

Every underlying script keeps track of what it has already written and skips it. So an
interrupted `--full` run, even one where this script was killed, is safe to re-run.

By default the output goes to **`full_run/<design>/<key>/`, not `data/`**. A fresh run is kept
separate from the shipped, already-reviewed results, so nothing gets overwritten by accident.
To report a fresh run, register its `results.csv` in the catalog (see below) and rebuild.
To regenerate the shipped results in place instead, pass `--out-dir data`.

This is slow and costs real money: full replay or 6-round free-run generation, plus
classification, for every combination. The default report-only mode above is what most
people want. `--combo` applies to `--full` only; the reports always cover the whole catalog.

## Adding or removing a result

`report/results_catalog.json` lists every results file the reports use, tagged with its
architecture, design and model; the reports group the files automatically. In the catalog the
architectures are named `single-baseline`, `single-personality`, `single-RAG` and `two-agent`.
Their data folders are `single`, `single_personality`, `single_rag` and `multi`.

```bash
python3 report/alignment_reports.py list            # what is registered, and the tables it yields
python3 report/alignment_reports.py add single-RAG replay gpt-5.4 data/replay/single_rag/results.csv
python3 report/alignment_reports.py remove single-RAG replay gpt-5.4
python3 report/alignment_reports.py build           # Kwon + Gao, all three comparison kinds
python3 report/alignment_reports.py build --style gao --by architecture --open   # one page
python3 report/meister_report.py                    # replay vs free run
```

A results CSV needs the columns `GameId, Sender, Text, Phase, pred_behavior`, which is what the
classifiers write. A table is built when its group has at least two rows.

## Layout

```
run_pipeline.py                   the one entry point (see Quickstart above)

data/
  human_data.csv                  the real human negotiation transcripts
  scores.csv                      per-player BATNA values (who got the outside-option change)
  personality_adjective.json      Big-Five trait -> adjective list, used by personality.py
  replay/<key>/, freerun/<key>/   generated data + results.csv, one dir per combo
                                  (single_rag*/: results.csv only -- its generation traces hold
                                  the retrieved human turns, so they are not redistributed)

generation/
  step1_calibration.py    human-only power/calibration check (no API key needed)
  step2_generate.py       free-running generation (any architecture × model)
  step5_replay.py         replay generation (same architectures/models)
  single_agent.py         the turn-grouped, calculator-backed single agent used by single_rag
  two_agents.py           the multi-agent planner + realizer
  personality.py          Big-Five trait sampling for single_personality
  llm_backends.py         provider abstraction: gpt-5.4 / claude-sonnet-4.6 / gemini-2.5-pro

classify/
  classify_replay.py      GPT-5.4 behavior classifier for replay output
  classify_freerun.py     same, for free-running output
  classifier_bridge.py    shared low-level classifier plumbing

stats/
  rag_align.py            shared alignment-metric primitives and the RAG retrieval
  kodis_stats.py          Kwon/KODIS-style gap engine (mean gaps and their tests; the human baseline)
  gao_style_table.py      distribution-level JSD + permutation tests, Gao Fig. 4 panels
  step6_paired.py         replay-only paired analysis (TOST equivalence, Gwet's AC1, …)

report/
  alignment_reports.py    Kwon + Gao reports by model / type / architecture -> output/
  meister_report.py       replay vs free run (Meister et al. 2025, Eq. 1), forest plot -> output/
  paper_figures.py        ACL figure export: printed-size SVG -> vector PDF, captions, LaTeX snippets
  results_catalog.json    which results.csv is which (architecture, design, model)

output/                   built reports and figures (git-ignored; rebuilt by run_pipeline.py)
full_run/                 run_pipeline.py --full's default output (git-ignored)
```

Run every script from the repo root (`python3 report/alignment_reports.py ...`,
`python3 generation/step2_generate.py ...`, etc.). `run_pipeline.py` already does this for you.
