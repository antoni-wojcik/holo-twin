# holo-twin

Differentiable digital twin of a Fraunhofer (far-field) holographic projector (SLM - Fourier lens - camera), and the code used to produce the results in "Digital twin for full field-of-view holographic patterning".

The model (`HoloSystem` in `src/twin/model.py`) learns the physical parameters of the system from pairs of (greyscale hologram, camera image). Each of the parameters is a separate, independently enable/disable-able module (see `ModuleFlags` in `src/twin/model.py`: `lut`, `slm_field`, `pixel`, `pupil`, `background`, `camera`):

1. Greyscale-to-phase look-up table (LUT)
2. Effective incident field on the SLM (beam profile, SLM curvature, dust)
3. Pixel crosstalk, fill factor and deadspace
4. Pupil aberrations
5. Stray light background
6. Camera affine transform and saturation

The physics and the training procedure are described in the Methods section of the paper and the Supplementary Information.

- Code: https://github.com/antoni-wojcik/holo-twin 
- Data and reference trained twins: https://doi.org/10.17863/CAM.134536

## Repository layout

| Path | Contents |
|---|---|
| `src/twin/` | The optical model (`HoloSystem`, `OpticsGeometry`, `ModuleFlags`) |
| `src/training/` | Staged training (camera → optics → background), ground-truth builder for simulation |
| `src/cgh/` | Hologram optimisation against a trained twin |
| `src/acquisition/` | Training patterns, hardware acquisition loop, loading acquired data |
| `src/hardware/` | `ISLM` / `ICamera` interfaces and an ideal (simulated) SLM and camera |
| `src/loss/`, `src/io/`, `src/utils/` | Losses, config/output handling, helpers |
| `pipelines/sim/` | Simulation only: build a random ground-truth system, fit a twin to it, run CGH |
| `pipelines/exp/` | Experiment: acquire data, fit a twin to it, run CGH and capture the result |
| `studies/` | The acquisition, training and CGH runs used in the paper |
| `analysis/` | Notebooks that make the paper figures |

Each pipeline script has a `DEFAULT_CONFIG` at the top and a `run(cfg)` function. To change a run, edit the config. Every run writes to its own folder `DATA_ROOT/<date>_<exp_base_name>/<exp_name>/`, with the `config.json` that produced it. `DATA_ROOT` is `data/` in the repo by default and can be changed with `DATA_DIR_ROOT` in a `.env` file at the repo root (see `src/config.py`).

## 1. System requirements

Tested on:
- Windows 10, Python 3.12.0
- PyTorch 2.13.0 + CUDA 12.6 (torchvision 0.28.0)
- NVIDIA GTX 1080 (8 GB VRAM); also run on RTX 3080 to RTX 5090 GPUs

Required Python packages (`pyproject.toml`): numpy, scipy, matplotlib, opencv-python, pytorch-msssim, tqdm, python-dotenv, plus PyTorch. Python ≥ 3.10. The full environment used for the paper, with exact versions, is in `requirements-lock.txt`. It includes Jupyter and hardware-control packages that are not needed to run the simulation or training.

Hardware:
- **Simulation** (`pipelines/sim/`) runs on CPU and needs no special hardware.
- **Training a twin on the experimental data** needs an NVIDIA GPU. The code was written to fit in 8 GB VRAM.
- **Acquisition and displaying holograms** (`pipelines/exp/acquire_data.py`, `pipelines/exp/run_cgh.py`) need the optical setup described in the Methods: HeNe laser (632.8 nm, Melles Griot 05-LHP-151), Santec SLM-200 phase-only LCoS SLM (1920 × 1200, 8 µm pitch), Thorlabs AC254-100-A Fourier lens (f = 100 mm), Thorlabs Kiralux CS126MU camera (3.45 µm pitch). Drivers for the SLM and camera are included:
  - Camera (`src/hardware/camera_thor/camera_cs126mu.py`): needs `pip install -e ".[vendor]"` (pylablib) and Thorlabs' camera software (ThorCam) installed.
  - SLM (`src/hardware/slm_santec/slm.py`, Windows only): copy `_slm_win.py`, `SLMFunc.dll` and `FTD3XX.dll` from the Santec SLM SDK into `src/hardware/slm_santec/`. These vendor files are not included in this repository.

  To use other hardware, write SLM and camera classes that implement `ISLM` (`src/hardware/islm.py`) and `ICamera` (`src/hardware/icamera.py`), and import them in place of the Santec/Thorlabs classes at the top of those two scripts.

## 2. Installation

```bash
git clone https://github.com/antoni-wojcik/holo-twin
cd holo-twin
python -m venv .venv
.venv\Scripts\activate            # Windows; on macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e .
```

Install PyTorch first, using the command for your system from [pytorch.org](https://pytorch.org/get-started/locally/) (the CPU-only build is enough for the simulation demo). `pyproject.toml` does not list `torch`, so that `pip install -e .` does not replace a CUDA build with a different one.

Optional extras: `pip install -e ".[vendor]"` for the camera driver (see Section 1), `pip install -e ".[mac]"` for SVG loading on macOS. On Windows, SVG targets are loaded with Inkscape (path set in `src/config.py`). Neither is needed for the demos.

Typical install time: 5–10 minutes, almost all of it downloading PyTorch.

## 3. Demo

### A. Simulation (CPU, no data needed)

```bash
python pipelines/sim/run_training.py
```

This builds a random ground-truth system on a small grid (160 × 200 SLM pixels), generates training data from it, and fits an independently initialised twin to it.

Expected output: a folder `data/<date>_sim_experiment/twin_fit/` containing `ground_truth/` (the true model and its report) and one folder per training stage (`Camera/`, `Optics/`, `Background/`), each with `twin_model.pt`, a model report, loss and convergence plots, and sample true/predicted camera images. The fitted parameters are compared against the ground truth at the end, and the predicted and true camera images should match closely.

Run time: roughly 5–10 minutes on a desktop CPU.

To then run CGH with the fitted twin, replace `<date>` in `twin_path` and `ground_truth_path` in `pipelines/sim/run_cgh.py` with the date of your training run folder, and run:

```bash
python pipelines/sim/run_cgh.py
```

Expected output: the optimised holograms, the target, and the image predicted by the twin and by the ground truth. Run time: a few minutes on CPU.

### B. Fitting a twin to the experimental data (GPU)

1. Download `ap_4pi_training_data` from https://doi.org/10.17863/CAM.134536 and put it in `data/` (or your `DATA_ROOT`).
2. In `pipelines/exp/run_training.py`, set `acquisition_path="ap_4pi_training_data"`.
3. Run:

```bash
python pipelines/exp/run_training.py
```

By default, this fits all modules to 4 checkerboard, 1 mask and 150 random-hologram captures, and evaluates on 10 held-out captures (alternative sizes can be specified and have been explored in the article).

Expected output: `data/<date>_twin_fit/run/` with one folder per stage, as in demo A, and the validation losses printed at the end. The result can be compared with the reference twin trained on the same data, provided in `twin_fits/4p_ap_fit/ap_4pi` in the dataset. The losses are described in the paper.

Run time on a GTX 1080: about 30 minutes (camera stage < 1 min, optics stage ~24 min, background stage ~5 min). On newer GPUs (RTX 3080 to RTX 5090), training and CGH take a few minutes.

## 4. Use on your own data

**Acquire data.** Implement `ISLM` and `ICamera` for your hardware (see Section 1), set the wavelength, exposures and number of samples in `pipelines/exp/acquire_data.py`, and run it. It writes:

```
checkers/   mask/   noise/   validation/
   each containing holos/<i>.npy and captures/<i>.npy
```

`holos/<i>.npy` are the displayed holograms as uint16 grey levels, 0 to `grayscale_range − 1`, at the full SLM size. `captures/<i>.npy` are the raw camera frames (uint16). Data from other sources can be used if it is saved in this layout.

**Train a twin.** In `pipelines/exp/run_training.py` set `acquisition_path`, `grayscale_range` (number of SLM grey levels, e.g. 1024 for a 10-bit SLM), and the `OpticsGeometry` for your system: `slm_pixels`, `slm_pixel_pitch`, `wavelength`, `focal_length`, `camera_shape` (the central camera region used for training), `camera_pixel_pitch` and `camera_exposure_time`. See `OpticsGeometry` in `src/twin/model.py` for all fields.

**Generate holograms.** In `pipelines/exp/run_cgh.py` set `twin_path` to the trained `twin_model.pt` and the target (`pattern`, `target_path`), then run it. Set `compute=False` and `holo_path` to redisplay holograms you already generated. CGH of a batch of 4 full-resolution (1920 × 1200) holograms with all modules enabled took about 6.5 minutes on a GTX 1080, and a few minutes or less on RTX 3080 to RTX 5090 GPUs.

## Reproducing the paper results

- `studies/`: the scripts that ran each acquisition, training and CGH study in the paper (see `studies/README.md`). The data they produced is in the dataset at https://doi.org/10.17863/CAM.134536.
- `analysis/analysis.ipynb`: figures in the main text.
- `analysis/sim_analysis.ipynb`: simulation figures in the Supplementary Information.

## License

MIT, see `LICENSE`.

## Citation

If you use this code, please cite the paper: "Digital twin for full field-of-view holographic patterning" (under review). Code: https://github.com/antoni-wojcik/holo-twin. Data: https://doi.org/10.17863/CAM.134536.
