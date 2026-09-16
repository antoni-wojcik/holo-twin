# studies/

Standalone scripts for the experiment sweeps, replacing the equivalent
notebook cells in `run_experiment.ipynb`. Each pipeline call (acquire /
train / cgh) runs in its own subprocess, so PyTorch/CUDA state can never
accumulate across runs the way it did in the notebook's single long-lived
kernel -- every run starts with a clean GPU, guaranteed, because it's a
brand-new process. See `runner.py`'s docstring for the full reasoning.

## Running a study

Each `study_XX_*.py` is runnable on its own, from the repo root, in the
same venv the notebook used (`pip install -e .`):

```
python -m studies.study_04_sample_sweep
```

It prints progress for each run as it happens (and tees the same output to
`studies/_runs/<study>/<run_id>/log.txt`), then a final summary. A failed
run doesn't stop the sweep -- the rest still run, same as the notebook's
try/except loops did.

## The unattended chain (studies 3 -> 4 -> 6)

```
python -m studies.run_main_sequence
```

Runs study 3 (acquire), feeds its output into study 4 (the sample-count
sweep), takes the 150-sample result, and feeds that into study 6 (the
5-target CGH sweep) -- all without needing to babysit it. Study 5
(background mask cleanup) is deliberately not part of this: it's a manual,
plot-driven step, run separately whenever convenient, and study 6 runs
with the background module off so it doesn't need study 5's mask anyway.

If this dies partway through (power cut, hardware fault, anything), just
re-run the same command -- `studies/_runs/manifest.jsonl` tracks every
run's outcome, so already-succeeded runs are skipped, not redone.

## Adding a new experiment

Copy the closest existing `study_XX_*.py` and edit the config diffs at the
top -- that's genuinely it. `study_07_module_ablation.py` is set up
specifically as a template for the "I'll add more of these by hand" case:
edit the `ABLATIONS` list and re-run.

Every study script follows the same shape:

1. Import `DEFAULT_CONFIG` from the relevant `pipelines.exp.*` module.
2. `build_configs()` returns `{run_id: config}`, built via
   `dataclasses.replace(DEFAULT_CONFIG, ...)` -- only the fields that
   differ from default need to be listed.
3. `if __name__ == "__main__": run_study(MODULE, STUDY, build_configs())`
   runs every config in the dict, one isolated subprocess at a time.

Chained studies (1, 2, and run_main_sequence.py) call `run_isolated()`
directly instead, so they can pass one step's output path into the next
step's config -- see `study_01_no_aperture_4pi_comparison.py` for the
pattern.

## Where things end up

- `studies/_runs/manifest.jsonl` -- one line per run attempt: study,
  run_id, status, output path, duration, log path. This is what makes
  resuming possible.
- `studies/_runs/<study>/<run_id>/` -- that run's `config.json` (what was
  actually passed to the worker), `result.json` (what it returned), and
  `log.txt` (its full stdout/stderr). This is separate from -- and not a
  replacement for -- the config/log copies each pipeline's own `ExpIO`
  already writes under `DATA_ROOT`; this copy exists purely as the
  hand-off mechanism into the subprocess and for debugging a failed run
  without digging through DATA_ROOT.
- `studies/_runs/` is gitignored, same as `data/`.

## What's NOT automated here

- **Study 5** (background field cleanup) stays exactly where it is in the
  notebook -- it needs a human looking at the mask plot and adjusting
  thresholds, which doesn't fit the "fresh subprocess" model (or any
  unattended model). Run it whenever, independent of everything above.
- **Study 7** is a hand-edited template, not a fixed sweep -- see above.
