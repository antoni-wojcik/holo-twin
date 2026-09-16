"""
Shared CGH pipeline: given a config, a (trained) twin, a target pattern,
and a way to "evaluate" a hologram, run cfg.batch_num independent CGH
batches and average them, saving everything - holograms, loss curve, the
config itself, and the actual vs. predicted images, per batch and averaged
- via ExpIO.
"""
import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from src.io.base_config import BaseConfig
from src.twin.model import HoloSystem, ModuleFlags
from src.cgh.optimize import generate_holograms
from src.training.plotting import plot_cgh_loss, render
from src.utils.utils import squish_image
from src.io.data_io import ExpIO, data_path


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class CGHConfig(BaseConfig):
    """
    Everything needed to run one CGH job, shared between hardware and
    simulation-only experiments. `run_cgh` takes this plus the handful of
    live objects that can't be serialized to JSON (model, target tensor,
    loss function, evaluate callback).
    """

    # --- experiment bookkeeping ---
    twin_path: str = ""              # path to the trained twin checkpoint, relative to DATA_ROOT
    exp_base_name: str = "cgh"
    exp_name: str = "run"
    exp_description: str = ""
    seed: int = 0

    # --- target pattern (resolved by the master script; kept here for the record and for sweeps) ---
    pattern: str = "circle"          # e.g. "circle", "checkerboard", "meta", "custom"
    target_path: Optional[str] = None  # image path, only used for image-based patterns (e.g. "custom")
    checkerboard_pitch: int = 10     # checkerboard square size in pixels; only used when pattern == "checkerboard"

    # --- optimisation (passed through to generate_holograms) ---
    compute: bool = True             # False -> load a previously computed hologram instead of optimizing
    holo_path: Optional[str] = None  # dir (relative to DATA_ROOT) containing a saved holograms.npy; required when compute=False
    batch_num: int = 1               # number of independent CGH batches to run and average over
    batch_size: int = 4
    micro_batch_size: int = 2
    num_epochs: int = 200
    lr: float = 5e-2
    weight_decay: float = 1e-4
    mask_centre: bool = True
    dark_penalty: bool = True        # penalize dark target regions lighting up; turn off for targets that are mostly midtone/dark (e.g. grayscale photos), where it distorts more than it helps
    lr_warmup_frac: Optional[float] = None  # None -> constant `lr` (old behaviour); else linear warmup to `lr` over this fraction of num_epochs, then cosine decay to 0 -- use if optimisation diverges late in a run

    # --- model preparation (applied to model_twin by run_cgh, in this order) ---
    # NOTE: `fov` is NOT applied here -- see run_cgh's docstring.
    module_flags: Optional[ModuleFlags] = None            # active/frozen overrides for model_twin
    remove_phase_ramp: bool = False
    background_aperture_mask_path: Optional[str] = None   # relative to DATA_ROOT; if set, mask model_twin.background.field to this aperture
    pixel_fast_approx: Optional[bool] = None               # if set, overrides model_twin.pixel.fast_approx
    clean_aperture: bool = True

    # --- evaluation ---
    fov: float = 1.0                        # multiples of the first diffraction order; see run_cgh docstring
    square_far_field: Optional[bool] = None  # None -> leave the twin's own loaded/trained setting; True/False -> force square/rectangular far-field sampling via HoloSystem.square_far_field() before building the target -- e.g. small checkerboard pitches alias badly on a non-square lattice
    exposure_time: Optional[float] = None   # exposure the actual capture was taken at; None = twin's own training exposure

    @property
    def effective_exposure_time(self) -> Optional[float]:
        """`exposure_time` scaled for `fov` (the same total power spreads
        over a larger area at larger FOV, so a real capture needs a longer
        exposure to stay comparable). None if `exposure_time` is unset."""
        if self.exposure_time is None:
            return None
        return self.exposure_time * self.fov ** 2

    def _apply_flags(self, model: HoloSystem) -> None:
        """Apply `module_flags` on top of `model`'s current active/frozen
        state. Only the fields explicitly set on `module_flags` override;
        everything else is left as-is."""
        if self.module_flags is not None:
            model.set_active(default=None, flags=self.module_flags)
            model.set_unfrozen(default=None, flags=self.module_flags)


@dataclass
class HardwareCGHConfig(CGHConfig):
    """CGHConfig plus the physical settings needed to display a hologram
    on the real SLM and capture it with the real camera."""
    wavelength: float = 633.0   # nm
    cam_gain: float = 0.0


@dataclass
class SimCGHConfig(CGHConfig):
    """CGHConfig plus the ground-truth model that stands in for real
    hardware in simulation-only CGH runs."""
    ground_truth_path: str = ""
    ground_truth_flags: Optional[ModuleFlags] = None   # active-state overrides for model_true

    def prepare_ground_truth(self, model_true: HoloSystem) -> None:
        """model_true is never optimized here, only forwarded through --
        freeze it, apply any active-state overrides, and set eval mode."""
        model_true.set_unfrozen(default=False)
        if self.ground_truth_flags is not None:
            model_true.set_active(default=None, flags=self.ground_truth_flags)
        model_true.eval()


# --------------------------------------------------------------------------
# Prediction (twin, or ground truth playing the same role)
# --------------------------------------------------------------------------

@torch.no_grad()
def cgh_predict_images(model: HoloSystem, g: torch.Tensor, camera_shape: tuple, exposure_time: Optional[float] = None):
    """
    Forward a hologram through `model` (twin or ground truth) to get its
    predicted far-field intensity and camera-plane image. Used both to get
    the twin's *predicted* images for comparison against an actual capture,
    and (in simulation) to get the ground-truth model's own camera-plane
    render, standing in for that actual capture in the first place.

    The camera affine warp + saturation are applied directly here via the
    camera module's private methods, independent of whether `model.camera`
    is itself active -- this always gives the camera-plane prediction, even
    when `model`'s own forward pass runs with the camera module disabled
    (as it typically is during CGH optimisation).

    Parameters
    ----------
    model : HoloSystem
        The model to predict with (twin or ground truth).
    g : torch.Tensor, shape (batch_size, P, Q)
        Hologram(s) to forward through the model.
    camera_shape : tuple
        Shape (H, W) to render the camera-plane image at -- e.g. the real
        camera's sensor shape, or the shape of whatever image this
        prediction will be compared against.
    exposure_time : float, optional
        Exposure time to scale the predicted intensity to, relative to
        `model.geometry.camera_exposure_time` (what the model was
        trained/calibrated at). None applies no scaling (ratio = 1).

    Returns
    -------
    I_mean_square : np.ndarray -- predicted far-field intensity, unsquished to a square image
    C_mean        : np.ndarray -- predicted camera-plane image
    """
    g = g.to(model.geometry.device).float()

    I_pred = model.get_intensity(model.propagate(g))
    if exposure_time is not None:
        I_pred = I_pred * (exposure_time / model.geometry.camera_exposure_time)

    I_mean = I_pred.mean(dim=0).cpu().numpy()
    img_max_dim = max(I_mean.shape)
    I_mean_square = squish_image(I_mean, (img_max_dim, img_max_dim))

    C_pred = model.camera._affine(I_pred, camera_shape)
    C_pred = model.camera._saturate(C_pred)

    C_mean = C_pred.mean(dim=0).cpu().numpy()

    return I_mean_square, C_mean


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def _prepare_model(cfg: CGHConfig, model: HoloSystem) -> None:
    """Put `model_twin` into the state CGH optimisation/evaluation expects.
    Freezes everything (only `g` is ever optimized), then layers on
    `cfg.module_flags` and the one-off preprocessing steps a CGH run wants.

    Does NOT touch `cfg.fov` -- see run_cgh's docstring for why."""
    model.eval()
    model.set_unfrozen(default=False)
    cfg._apply_flags(model)

    if cfg.remove_phase_ramp:
        model.slm_field.remove_ramp()
    if cfg.background_aperture_mask_path is not None:
        _mask_background_aperture(model, cfg.background_aperture_mask_path)
    if cfg.pixel_fast_approx is not None:
        model.pixel.fast_approx = cfg.pixel_fast_approx
    if cfg.clean_aperture:
        model.slm_field.clean_aperture()


def _mask_background_aperture(model: HoloSystem, mask_path: str) -> None:
    """Restrict the background field to a pre-determined aperture region,
    so CGH only "sees" stray light inside the SLM's active area.
    `mask_path` is relative to DATA_ROOT, resolved here."""
    mask = torch.load(data_path(mask_path))
    field = model.background.field.clone() * mask
    field[mask == 0] = 0 + 0j
    model.background.field = field


def run_cgh(
    cfg: CGHConfig,
    model_twin: HoloSystem,
    target: torch.Tensor,
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    exp: Optional[ExpIO] = None,
    evaluate_fn: Optional[Callable[[torch.Tensor, int, Optional[ExpIO]], np.ndarray]] = None,
):
    """
    Run cfg.batch_num independent CGH batches, one after another, then
    average them. Prepares the model once, then per batch: optimise
    (compute=True) or load (compute=False) a hologram, evaluate it if
    `evaluate_fn` is given, and save everything -- holograms, loss curve,
    the config, and (if evaluated) that batch's own true/predicted images
    -- into its own nested "<exp_name>/batch_<i>" ExpIO run. Finally,
    averages every batch's true/predicted images together and saves the
    average into `exp`'s own directory.

    evaluate_fn(g, batch_idx, exp) -> np.ndarray is called once per batch
    with that batch's hologram, its index, and its own nested ExpIO run
    (None if `exp` itself is None) -- e.g. real hardware display + camera
    capture in experiments, or the ground-truth model's predicted camera
    image in simulation.

    Returns (avg_true, avg_pred_I, avg_pred_C) -- the batch-averaged true
    and predicted images -- or None if no evaluate_fn was given.
    """
    expected_shape = tuple(model_twin.geometry.far_fov_samples)
    if tuple(target.shape) != expected_shape:
        raise ValueError(
            f"target shape {tuple(target.shape)} does not match "
            f"model_twin.geometry.far_fov_samples {expected_shape} at fov={cfg.fov} -- "
            f"did you call model_twin.set_fov(cfg.fov) before building the target?"
        )

    _prepare_model(cfg, model_twin)
    render(model_twin.report(), "model_report", exp=exp)

    # Resolve the experiment name to a relative path for nested batch runs
    resolved_exp_name = None
    if exp is not None:
        resolved_exp_name = str(Path(*Path(exp.relative_path).parts[1:]))

    img_max_dim = max(target.shape)
    avg_true = avg_pred_I = avg_pred_C = None

    for i in range(cfg.batch_num):
        batch_exp = None
        if exp is not None:
            batch_exp = ExpIO(base_name=cfg.exp_base_name, exp_name=f"{resolved_exp_name}/batch_{i}", description=cfg.exp_description)
            cfg.save(batch_exp.get_new_path("config", "json"))

        # Distinct-but-deterministic seed per batch
        torch.manual_seed(cfg.seed + i)

        if cfg.compute:
            g, loss_history = generate_holograms(
                model_twin, target, loss_func,
                num_epochs=cfg.num_epochs, micro_batch_size=cfg.micro_batch_size,
                batch_size=cfg.batch_size, lr=cfg.lr, weight_decay=cfg.weight_decay,
                mask_centre=cfg.mask_centre, dark_penalty=cfg.dark_penalty,
                lr_warmup_frac=cfg.lr_warmup_frac,
            )
            if batch_exp is not None:
                batch_exp.save_npy(g.cpu().numpy(), "holograms")
                batch_exp.save_npy(loss_history, "loss_history")
            render(plot_cgh_loss(loss_history, "Hologram Generation Loss"), "loss_history", exp=batch_exp)
        else:
            if cfg.holo_path is None:
                raise ValueError("cfg.holo_path must be set when cfg.compute=False")
            # "<holo_path>/batch_<i>/holograms.npy" -- each batch redisplays
            # the hologram that batch itself produced originally.
            holo_path = os.path.join(data_path(cfg.holo_path), f"batch_{i}", "holograms.npy")
            g = torch.from_numpy(np.load(holo_path)).float().to(model_twin.geometry.device)

        # Return whatever's cached-but-idle to the driver before the next
        # batch, so allocations don't fragment across batches.
        torch.cuda.empty_cache()
        gc.collect()

        if evaluate_fn is None:
            continue

        C_true = np.asarray(evaluate_fn(g, i, batch_exp))
        I_pred, C_pred = cgh_predict_images(
            model_twin, g, camera_shape=C_true.shape, exposure_time=cfg.effective_exposure_time,
        )

        if batch_exp is not None:
            batch_exp.save_npy(C_true, "capture_true")
            batch_exp.save_npy(I_pred, "intensity_pred")
            batch_exp.save_npy(C_pred, "capture_pred")

        avg_true = C_true if avg_true is None else avg_true + C_true
        avg_pred_I = I_pred if avg_pred_I is None else avg_pred_I + I_pred
        avg_pred_C = C_pred if avg_pred_C is None else avg_pred_C + C_pred

    if evaluate_fn is None or exp is None:
        return None

    avg_true = avg_true / cfg.batch_num
    avg_pred_I = avg_pred_I / cfg.batch_num
    avg_pred_C = avg_pred_C / cfg.batch_num

    exp.save_image(avg_true, "avg_capture")
    exp.save_image(avg_pred_I, "avg_intensity_predicted")
    exp.save_image(avg_pred_C, "avg_capture_predicted")

    target_unsquished = squish_image(target.cpu().numpy(), (img_max_dim, img_max_dim))
    exp.save_image(target_unsquished, "target")

    # Align the true capture to the target grid for a direct pixel-wise comparison
    # (camera module holds the far-field -> camera-plane affine; we need its inverse).
    avg_true_t = torch.from_numpy(avg_true).float().to(model_twin.geometry.device).unsqueeze(0)
    avg_true_aligned = model_twin.camera.affine_inverse(avg_true_t).squeeze(0).detach().cpu().numpy()
    exp.save_image(avg_true_aligned, "avg_capture_affine")

    return avg_true, avg_pred_I, avg_pred_C