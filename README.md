# holo-twin

A differentiable digital twin of a Fraunhofer (far-field) holographic optical system (SLM - free-space propagation - camera), implemented in PyTorch. The model itself lives in `src/twin/model.py` (`HoloSystem`, built from an `OpticsGeometry`) and learns the system's physical parameters directly from (greyscale hologram, monochrome camera image) pairs, with only the initial optical geometry estimate needed:

1. Greyscale voltage-phase look-up table (LUT) response
2. Effective incident field (laser beam profile + SLM curvature + dust)
3. Pixel crosstalk, fill factor and deadspace effects
4. Pupil aberrations (Seidel + tip/tilt)
5. Stray light background field
6. Camera affine transform and saturation response

Each of these is a separate, independently enable/disable-able module (see `ModuleFlags` in `src/twin/model.py`: `lut`, `slm_field`, `pixel`, `pupil`, `background`, `camera`). The physics is described in the paper; this repo is the implementation.

Everything else in the repository is training and experiment code built around that model: capturing real hardware data, training the twin against it (or against a known simulated ground truth), and generating/projecting holograms with a trained twin - this is what produced the paper's results and figures.

## Layout

| Path | What it is |
|---|---|
| `src/twin/` | The optical model itself (`HoloSystem`), built from modular per-effect submodules |
| `src/training/` | The training pipeline: stages (camera - optics - background), losses, convergence plotting |
| `src/acquisition/` | Hardware data capture: patterns, acquisition loop |
| `src/cgh/` | Hologram generation/projection using a trained twin |
| `src/hardware/` | SLM/camera driver interfaces, plus vendor implementations (Santec, Thorlabs) |
| `src/loss/` | Loss functions and regularisation |
| `src/io/` | Config save/load and experiment output handling (`ExpIO`) |
| `pipelines/exp/` | Experimental workflows: acquire real data, train a twin on it, run CGH |
| `pipelines/sim/` | Simulation-only counterparts: build a randomized ground-truth twin, fit an independently-initialized twin against it, and run/evaluate CGH against that known ground truth |

Each pipeline script is a thin, self-contained entry point: a `DEFAULT_CONFIG` at the top and a `run(cfg)` function underneath that does the work. To change what a run does, change the config values passed to `run()` - directly, or via
`dataclasses.replace(DEFAULT_CONFIG, ...)` - the pipeline body itself shouldn't need to change. This is what makes it straightforward to run several configs in a loop, or chain pipelines together (e.g. `run_experiment.ipynb`, which chains acquisition - training - CGH end-to-end), without touching pipeline code.

A single `train_twin()` routine (in `src/training/`) drives all staged training - camera, then optics, then background - with the same routine used for both simulation and real data; only the data source and loss masking differ. Each stage uses a cosine warmup/decay LR schedule per parameter group.

CGH (`src/cgh/`) optimizes a greyscale hologram (or batch, for time-multiplexed display) against a trained twin to reproduce a target intensity pattern. The same optimize-or-reload logic is used in both pipelines, so a previously-generated hologram can be redisplayed (e.g. at a different exposure) without recomputing it - see `cfg.compute=False` +
`cfg.holo_path` in the CGH configs.

## Config and output

Every pipeline's config is a dataclass (`TrainingConfig`, `AcquisitionConfig`, etc., subclassing `BaseConfig` in `src/io/base_config.py`), saved and loaded as plain JSON.

Each run gets its own output directory via `ExpIO` (`src/io/data_io.py`), rooted at `DATA_ROOT`: `DATA_ROOT/<date>_<experiment-base-name>/<run-name>/`. The first run of a given name gets that bare name; re-running with the same name appends `_1`, `_2`, ... rather than overwriting. Every run directory contains the `config.json` that produced it, alongside whatever else it saved - model checkpoints, loss/convergence plots, and (for training runs) per-stage field maps and reports. A downstream pipeline can pick up an upstream run's output by pointing a config field at its path relative to `DATA_ROOT` (e.g. an `ExpTrainingConfig.acquisition_path` pointing at an acquisition run, or a CGH config's `twin_path` pointing at a training run's checkpoint).

## Install

Requires Python >=3.10.

**1. Clone the repo and create a virtual environment:**

```bash
git clone <repo-url>
cd holo-twin
python -m venv .venv
```

Activate it:
- Windows (PowerShell): `.venv\Scripts\Activate.ps1`
- Windows (cmd): `.venv\Scripts\activate.bat`
- macOS/Linux: `source .venv/bin/activate`

**2. Install PyTorch yourself, matched to your hardware, *before* installing this package:**

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

(use whichever command matches your setup from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) - `cu126` above assumes a CUDA 12.6-compatible driver; check with `nvidia-smi` if you're not sure. Use the CPU-only command if you don't have a CUDA GPU.)

`pyproject.toml` deliberately does **not** list `torch` as a dependency - it ships as several different builds (CPU-only, or CUDA 11.8/12.1/12.6/etc.), and `pip` has no way to know which one matches your driver. Installing it yourself first, in its own step, avoids `pip` silently reinstalling it with the wrong build later (e.g. replacing a working CUDA install with a CPU-only one).

**3. Install this package:**

```bash
python -m pip install -e .
```

This installs everything under `dependencies` in `pyproject.toml` plus this repo itself in editable mode, so `import src...` works from any script, notebook, or terminal in this environment - no `PYTHONPATH` configuration needed.

If you would rather not install it as a package, `pip install -r requirements.txt` (after installing torch as above) and run scripts from the repo root instead - `requirements.txt` excludes torch for the same reason.

### Hardware / vendor extras (optional)

Running the experimental acquisition/CGH pipelines (`pipelines/exp/`) against real hardware needs the vendor SLM and camera libraries (Santec SLM, Thorlabs camera) - these aren't on PyPI, so install them via the `vendor` extra:

```bash
python -m pip install -e ".[vendor]"
```

(the quotes matter in bash/zsh - without them, `[vendor]` is interpreted as a shell glob pattern rather than passed to pip.) If a vendor SDK isn't pip-installable at all (just a folder the vendor ships, no PyPI package or wheel), add its path to `PYTHONPATH` via your local `.env` file instead of forcing it into `pyproject.toml` - see Environment below.

Everything under `src/twin`, `src/training`, `src/cgh`, and `src/loss` - i.e. simulation, training, and CGH against a saved twin - works without the vendor extras, so simulation-only work doesn't need the hardware set up.

### SVG handling (platform-specific)

Loading SVG target images (`load_svg` in `src/io/img_handler.py`) needs an SVG rasterizer, and which one is used depends on your OS:

- **Windows**: `cairosvg` is difficult to install here (it needs native Cairo libraries
  with no simple pip wheel on Windows), so Inkscape is the default SVG backend. Install it separately from [inkscape.org](https://inkscape.org/) and make sure it's on your `PATH`.
- **macOS**: `cairosvg` installs cleanly via pip. Install it with the `mac` extra:
  `python -m pip install -e ".[mac]"`.

`load_svg` picks the right backend automatically based on your platform - you just need the corresponding tool installed first. `config.py` (see Environment below) resolves the Inkscape path and checks whether `cairosvg` is available; this is also what the paper-figure-generation code uses to convert SVG panels to PDF.

## Environment

Machine-specific settings (where output goes, paths to the external tools above) are centralised in `config.py`, the one place in the project that reads `os.environ` directly - everything else imports the resolved values from it. It loads a `.env` file at the repo root via python-dotenv:

```bash
cp .env.example .env
# then edit .env
```

falling back to a real environment variable if `.env` doesn't set one, then to a sensible default.

The setting that matters day-to-day is the output root: set `DATA_DIR_ROOT` in `.env` to where you want experiment output written - `config.py` exposes this as `DATA_ROOT`, which `ExpIO` builds every run's output directory under (see "Config and output" above). If unset, it defaults to a `data/` folder at the repo root. `config.py` also resolves an Inkscape binary path (`INSKAPE_PATH`) and exposes `CAIROSVG_AVAILABLE`, both used by the SVG handling and paper-figure-generation code described above.

## Quickstart

**Simulation** - build a random ground truth, fit a twin against it, and run CGH:

```bash
python pipelines/sim/run_training.py
python pipelines/sim/run_cgh.py
```

Each script's `DEFAULT_CONFIG` controls what's run - for `run_training.py`, this includes which physical effects exist in the simulated ground truth and which twin parameters are held fixed, useful for isolating which module is responsible when convergence fails. Edit `DEFAULT_CONFIG` directly, or build a variant with `dataclasses.replace(DEFAULT_CONFIG, ...)` and pass it to `run()` yourself (e.g. in a notebook or a sweep script).

**Experimental** - acquire hardware data, then fit and evaluate a twin against it:

```bash
python pipelines/exp/acquire_data.py
python pipelines/exp/run_training.py
python pipelines/exp/run_cgh.py
```

As above, each script runs its own `DEFAULT_CONFIG` when run directly. To chain them - feeding an acquisition run's path into training, and a training run's checkpoint into CGH - build each config from the previous step's returned path rather than re-running with defaults; `run_experiment.ipynb` has a worked end-to-end example of this.

Every run creates a dated, numbered folder under the data root (via `ExpIO`) containing model checkpoints, loss curves, model reports, and field maps for each training stage,
plus the raw acquired/generated data.

## Troubleshooting

**`pip install -e .` (or some other install step) broke my PyTorch - `torch.cuda.is_available()` is suddenly `False`, or training that used to run on GPU now runs on CPU.**

This means `torch` got reinstalled generically at some point instead of via the CUDA-matched command in step 2 of Install - e.g. `torch` was listed as a plain dependency somewhere, or a `pip install` in this environment pulled it in as a side effect of resolving some other package. Don't try to patch this in place; rebuild the environment from scratch:

```bash
deactivate
rm -rf .venv                    # PowerShell: Remove-Item -Recurse -Force .venv
python -m venv .venv
.venv\Scripts\Activate.ps1      # or the activation command for your shell
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e .
```

Check `nvidia-smi` for the CUDA version your driver actually supports and use the matching `--index-url` from pytorch.org's install page - don't assume `cu126` is correct for your machine.

**General rule:** nothing in this environment should ever reinstall `torch` implicitly. If you add a new dependency later that happens to depend on `torch`, check what version/build it wants before installing it here, or test it in a throwaway venv first.
