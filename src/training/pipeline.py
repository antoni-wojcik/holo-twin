"""
Shared twin-training pipeline.

`run_training()` is the single entry point called identically by
pipelines/exp/run_training.py and pipelines/sim/run_training.py: build (or
load) the twin, compute the ZOD + saturation masks, build the loss recipe
(identical between sim and experimental training), then hand off to
`train_twin()` for the actual 3-stage (camera -> optics -> background) run.
Everything that must be the same across sim and experimental training
lives here; the pipeline scripts only differ in how they build the
`DataSource`s they pass in (real captures vs. a live ground-truth model).

`train_twin()` itself is unchanged from before: it just runs the 3 stages
in order, given a fully-prepared model and losses. It used to be called
directly by each pipeline script; now it's called by `run_training()`
below, which is what builds the model/masks/losses it needs.
"""
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from src.training.stage import (
    ParamGroup,
    OpticsStage, BackgroundStage,
    CameraEstimateCVStage,
    compare_prediction
)
from src.training.train import LossSpec, DataSource, LossSuite, evaluate_losses
from src.training.build_twin import build_twin
from src.training.build_true import GroundTruthConfig
from src.training.masks import get_zod_mask_checker, get_zod_mask_median
from src.training.plotting import render, render_image
from src.loss.losses import build_ssim_loss, build_ms_ssim_loss, mse_loss
from src.loss.reg import total_variation_reg
from src.twin.model import HoloSystem, ModuleFlags, OpticsGeometry
from src.io.base_config import BaseConfig
from src.io.data_io import ExpIO, data_path


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class TrainingConfig(BaseConfig):
    """
    Everything needed to train a digital twin against a fixed dataset of
    (hologram, camera-image) pairs -- shared between experimental and
    simulation training. Subclassed as ExpTrainingConfig and
    SimTrainingConfig for the one thing that actually differs between
    them: where the training data comes from (see each subclass's
    docstring).
    """

    # --- experiment bookkeeping ---
    exp_base_name: str = "twin_fit"
    exp_name: str = "run"
    exp_description: str = ""
    seed: int = 0   # torch + numpy seed, set before building/loading the twin -- for repeatable training runs

    # --- physical geometry the twin is built/trained at ---
    geometry: OpticsGeometry = field(default_factory=lambda: OpticsGeometry(
        slm_pixels=(1200, 1920), scale=2, camera_shape=(2000, 2000), camera_exposure_time=0.2, device="cuda",
    ))

    # --- twin initialisation: load an existing checkpoint, or build fresh with these flags ---
    load_twin_path: Optional[str] = None   # relative to DATA_ROOT; if set, build_twin() below is skipped entirely
    init_lut_scale: float = 4 * np.pi
    slm_field_divisor: Optional[int] = None   # if set, the SLM field is downsampled by this factor after build_twin()
    init_crosstalk_sigma: tuple = (0.3, 0.3)
    init_slm_field_scale: float = 1.0
    num_pupil_tiles: int = 40
    init_deadspace_reflectance: float = 0.8
    use_crosstalk_fast_approx: bool = False
    use_crosstalk_residual: bool = True
    use_pupil_vignetting: bool = False
    use_lut_depth: bool = False

    # --- what to train ---
    run_stages: tuple = (True, True, True)        # (camera, optics, background) -- False skips a stage entirely
    module_flags: Optional[ModuleFlags] = None     # active/frozen overrides applied on top of each stage's own defaults
    fix_camera: bool = False                       # freeze the camera's affine params during the optics stage
    mask_method: str = "checker"                   # "checker" or "median", see run_training()

    # --- data: how many samples to actually train/validate on, independent of how many exist on disk ---
    num_checker_samples: int = 4
    num_noise_samples: int = 150
    num_validation_samples: int = 10
    grayscale_range: int = 1024   # quantization levels stored holograms were saved at -- must match the
                                   # acquisition that produced them (a real SLM's native range for ExpTrainingConfig)
    skip_eval_losses: bool = False   # if True, no "structure" eval spec is built at all -- no per-batch eval
                                      # panel, no per-epoch validation tracking, no final validation summary.
                                      # Pure speed switch for when even 10 validation samples is too slow.

    # --- optimisation ---
    mask_threshold: float = 0.1   # threshold for ZOD mask generation (see get_zod_mask_checker/median)
    batch_size: int = 4
    micro_batch_size: int = 1
    iterations: tuple = (200, 100)   # (optics epochs, background epochs) -- the camera stage always runs for 1

    # --- regularisation, see run_training()'s full_loss/back_loss ---
    tv_lambda_amp_optics: float = 1e-3
    tv_lambda_phase_optics: float = 1e-3
    tv_lambda_l2_optics: float = 1e-4
    tv_lambda_amp_background: float = 5e-2
    tv_lambda_phase_background: float = 0.0
    tv_lambda_l2_background: float = 1e-4


@dataclass
class ExpTrainingConfig(TrainingConfig):
    """TrainingConfig plus the path to a real hardware acquisition run (see pipelines/exp/acquire_data.py)."""
    acquisition_path: str = ""   # relative to DATA_ROOT; must contain checkers/mask/noise/validation subdirs


@dataclass
class SimTrainingConfig(TrainingConfig):
    """
    TrainingConfig plus the ground-truth parameters used to build a fresh
    HoloSystem each run and generate training data from it on the fly
    (via make_sim_batch_fn) -- nothing is loaded from disk.
    """
    ground_truth: GroundTruthConfig = field(default_factory=GroundTruthConfig)


# --------------------------------------------------------------------------
# Shared training entry point
# --------------------------------------------------------------------------

def run_training(
    cfg: TrainingConfig,
    data_checker: DataSource,
    data_random: DataSource,
    data_mask: DataSource,
    exp: Optional[ExpIO] = None,
    data_validation: Optional[DataSource] = None,
    model_true: Optional[HoloSystem] = None,
) -> Tuple[HoloSystem, Optional[dict]]:
    """
    Build (or load) the twin, compute the ZOD + saturation masks, build
    the loss recipe, and run the 3-stage train_twin() pipeline. Identical
    between experimental and simulation training -- the pipeline scripts
    only differ in how they build the DataSources passed in here.

    Parameters
    ----------
    cfg : TrainingConfig
        All serializable training settings.
    data_checker, data_random : DataSource
        Grating-probe and random-hologram training data (see src/acquisition/data.py).
    data_mask : DataSource
        Single-sample (1,1)-pitch ZOD probe, used when cfg.mask_method == "checker".
    exp : ExpIO, optional
        If given, the config, masks, per-stage checkpoints/losses, and
        validation/comparison results are all saved through it.
    data_validation : DataSource, optional
        Held-out random-hologram data (e.g. an acquisition's "validation"
        subdir) -- never trained on, evaluated once after training. If
        None, no held-out evaluation is performed.
    model_true : HoloSystem, optional
        Ground-truth model, if known (simulation only). If given, a
        retrieved-vs-true parameter comparison figure is rendered/saved
        after training via HoloSystem.report_diff().

    Returns
    -------
    model_twin : HoloSystem
        The trained twin.
    validation_losses : dict or None
        {"optics: <name>": float, ...} and {"background: <name>": float, ...},
        one prefixed sub-dict's worth of entries per stage that still had an
        eval spec, each averaged over data_validation -- see the comment
        above that block for why optics and background are evaluated (and
        prefixed) separately rather than merged into one evaluate_losses()
        call. None if data_validation was not given, or if cfg.skip_eval_losses
        left both stages' eval spec lists empty.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if exp is not None:
        cfg.save(exp.get_new_path("config", "json"))

    # ------------------------------------------------------------------
    # Twin: load a checkpoint if given, else build a fresh noisy init
    # ------------------------------------------------------------------
    if cfg.load_twin_path is not None:
        model_twin = HoloSystem.load(data_path(cfg.load_twin_path))
        model_twin.slm_field.clean_aperture()
    else:
        model_twin = build_twin(
            cfg.geometry,
            init_lut_scale=cfg.init_lut_scale, init_crosstalk_sigma=cfg.init_crosstalk_sigma,
            init_slm_field_scale=cfg.init_slm_field_scale,
            num_pupil_tiles=cfg.num_pupil_tiles, init_deadspace_reflectance=cfg.init_deadspace_reflectance,
            use_lut_depth=cfg.use_lut_depth, use_crosstalk_fast_approx=cfg.use_crosstalk_fast_approx,
            use_crosstalk_residual=cfg.use_crosstalk_residual, use_pupil_vignetting=cfg.use_pupil_vignetting,
        )

        if cfg.slm_field_divisor is not None:
            shape = (cfg.geometry.slm_pixels[0] // cfg.slm_field_divisor, cfg.geometry.slm_pixels[1] // cfg.slm_field_divisor)
            model_twin.slm_field.set_shape(shape, retain_field=True)

    # ------------------------------------------------------------------
    # ZOD mask: separates the static zeroth-order-deflected (ZOD) region
    # from the field actually controlled by the hologram, so it can be
    # excluded from the structure loss below.
    # ------------------------------------------------------------------
    if cfg.mask_method == "checker":
        _, C_mask = data_mask.batch_fn(torch.arange(1))
        zod_mask = get_zod_mask_checker(C_mask, threshold=cfg.mask_threshold)
    elif cfg.mask_method == "median":
        _, C_all = data_random.batch_fn(torch.arange(cfg.num_noise_samples))
        zod_mask = get_zod_mask_median(C_all, threshold=cfg.mask_threshold)
    else:
        raise ValueError(f"Unknown mask_method {cfg.mask_method!r}; use 'checker' or 'median'.")
    render_image(zod_mask, cmap="gray", name="ZOD Mask", exp=exp)

    # ------------------------------------------------------------------
    # Saturation mask: True only where a pixel is saturated (>=0.99) in
    # EVERY sample of the noise batch -- used to exclude permanently
    # blown-out camera pixels from the background loss below.
    # ------------------------------------------------------------------
    saturation_mask = torch.ones(cfg.geometry.camera_shape, dtype=torch.bool, device=cfg.geometry.device)
    num_batches = cfg.num_noise_samples // cfg.batch_size
    for i in range(num_batches):
        _, C_batch = data_random.batch_fn(torch.arange(i * cfg.batch_size, (i + 1) * cfg.batch_size))
        # True ONLY IF saturated in every image of this batch AND all previous batches
        saturation_mask &= (C_batch >= 0.99).all(dim=0)
    saturation_mask = (~saturation_mask).float()  # invert: 0 where always-saturated, 1 elsewhere
    render_image(saturation_mask, cmap="gray", name="Saturation Mask", exp=exp)

    # ------------------------------------------------------------------
    # Loss functions -- identical recipe for sim and experimental training.
    # ------------------------------------------------------------------
    ms_ssim_mask = build_ms_ssim_loss(mask=zod_mask)
    mse_mask = lambda I, T: mse_loss(I, T, mask=zod_mask)
    ssim_sat = build_ssim_loss(mask=saturation_mask)
    mse_sat = lambda I, T: mse_loss(I, T, mask=saturation_mask)
    structure_mask = lambda I, T: (ms_ssim_mask(I, T) + mse_mask(I, T)) * 0.5
    structure_sat = lambda I, T: (ssim_sat(I, T) + mse_sat(I, T)) * 0.5

    def optics_loss(I, T):
        """Loss for the optics stage: structure (ZOD mask) + SLM-field TV/L2 regularisation."""
        reg = total_variation_reg(
            field=model_twin.slm_field.field,
            lambda_tv_amp=cfg.tv_lambda_amp_optics, lambda_tv_phase=cfg.tv_lambda_phase_optics,
            lambda_l2=cfg.tv_lambda_l2_optics,
        )
        return structure_mask(I, T) + reg

    def back_loss(I, T):
        """Full loss for the background stage: structure (saturation mask) + background-field TV/L2 regularisation."""
        field_slm = torch.fft.ifftshift(
            torch.fft.ifft2(torch.fft.fftshift(model_twin.background.field), norm="ortho")
        )
        reg = total_variation_reg(
            field=field_slm,
            lambda_tv_amp=cfg.tv_lambda_amp_background, lambda_tv_phase=cfg.tv_lambda_phase_background,
            lambda_l2=cfg.tv_lambda_l2_background,
        )
        return structure_sat(I, T) + reg

    optics_loss_spec = LossSpec(optics_loss, name="structure (mask) + TV reg")
    back_loss_spec = LossSpec(back_loss, name="structure (sat mask) + TV reg")
    optics_eval_losses = [] if cfg.skip_eval_losses else [
        LossSpec(structure_mask, name="structure"),
    ]
    back_eval_losses = [] if cfg.skip_eval_losses else [
        LossSpec(structure_sat, name="structure"),
    ]

    # ------------------------------------------------------------------
    # Training exposure stats: how many times each individual training
    # pair is drawn over a full run, given train_model()'s per-epoch
    # sampling 
    # ------------------------------------------------------------------
    it_optics, it_back = cfg.iterations
    training_stats = {
        "exposures_per_sample_optics": exposures_per_sample(it_optics, cfg.batch_size, cfg.num_noise_samples),
        "exposures_per_sample_background": exposures_per_sample(it_back, cfg.batch_size, cfg.num_noise_samples),
    }
    print("Training exposure stats:", training_stats)
    if exp is not None:
        exp.save_npy(training_stats, "training_stats")

    # ------------------------------------------------------------------
    # Train. Each stage renders and saves its own convergence plot as it
    # finishes (see TrainingStage._report() in stage.py)
    # ------------------------------------------------------------------
    train_twin(
        model_twin, data_checker, data_random,
        optics_loss=optics_loss_spec, background_loss=back_loss_spec, sat_mask=saturation_mask,
        optics_eval_losses=optics_eval_losses, back_eval_losses=back_eval_losses,
        data_validation=data_validation,
        batch_size=cfg.batch_size, micro_batch_size=cfg.micro_batch_size,
        fix_camera=cfg.fix_camera, exp=exp, iterations=cfg.iterations,
        module_flags=cfg.module_flags, run_stages=cfg.run_stages,
    )

    # ------------------------------------------------------------------
    # Held-out validation: never trained on. The per-epoch mean/SEM already tracked into
    # each stage's convergence plot gives the training-time curve.
    # ------------------------------------------------------------------
    validation_losses = None
    if data_validation is not None and (optics_eval_losses or back_eval_losses):
        validation_losses = {}
        if optics_eval_losses:
            optics_validation = evaluate_losses(
                model_twin, data_validation, optics_eval_losses,
                batch_size=cfg.batch_size, micro_batch_size=cfg.micro_batch_size,
            )
            validation_losses.update({f"optics: {k}": v for k, v in optics_validation.items()})
        if back_eval_losses:
            background_validation = evaluate_losses(
                model_twin, data_validation, back_eval_losses,
                batch_size=cfg.batch_size, micro_batch_size=cfg.micro_batch_size,
            )
            validation_losses.update({f"background: {k}": v for k, v in background_validation.items()})
        print("Validation losses:", validation_losses)
        if exp is not None:
            exp.save_npy(validation_losses, "validation_losses")

    # ------------------------------------------------------------------
    # Ground-truth comparison (simulation only, when the ground truth is known)
    # ------------------------------------------------------------------
    if model_true is not None:
        render(model_twin.report_diff(model_true), "model_report_diff", exp=exp)

    return model_twin, validation_losses


def exposures_per_sample(num_epochs: int, batch_size: int, num_training_samples: int) -> float:
    """
    Expected number of times each individual training pair is drawn over a
    full run, given train_model()'s per-epoch sampling (a fresh random
    batch_size-sample subset of num_training_samples every "epoch", not a
    full pass -- see train_model()'s docstring). In expectation this is
    (num_epochs * batch_size) / num_training_samples, since every epoch
    draws batch_size samples effectively uniformly at random from the pool.
    """
    return (num_epochs * batch_size) / num_training_samples


# --------------------------------------------------------------------------
# 3-stage training loop (camera -> optics -> background)
# --------------------------------------------------------------------------

def train_twin(
    model_twin: HoloSystem,
    data_checker: DataSource,
    data_random: DataSource,
    optics_loss: LossSpec,
    background_loss: LossSpec,
    sat_mask: torch.Tensor,
    optics_eval_losses: Optional[List[LossSpec]] = None,
    back_eval_losses: Optional[List[LossSpec]] = None,
    data_validation: Optional[DataSource] = None,
    batch_size: int = 4,
    micro_batch_size: Optional[int] = None,
    fix_camera: bool = False,
    exp: Optional[ExpIO] = None,
    iterations: tuple = (200, 100),
    module_flags: Optional[ModuleFlags] = None,
    run_stages: tuple = (True, True, True)
) -> dict:
    """
    Runs the 3 training stages (camera -> optics -> background) on an
    already-prepared model, given fully-built losses/masks. Called by
    run_training() above, which is what builds the model, masks, and
    losses this needs -- see that function's docstring for the shared
    recipe both experimental and simulation training use.

    Each stage renders and saves its own convergence plot as it finishes
    (see TrainingStage._report() in stage.py); there is no combined
    end-of-run plot.

    Returns
    -------
    histories : dict
        {"camera": history, "optics": history, "background": history}, one
        entry per stage that actually ran, each exactly as returned by
        train_model() (see TrainingStage.run()). A stage that was skipped
        has no entry. Purely for callers that want to re-plot the curves
        themselves -- training is unaffected by whether this is used.
    """
    it_optics, it_back = iterations
    histories = {}

    # Save the initial model state before training
    if exp is not None:
        subdir_name = "initial"
        model_twin.save(exp.get_new_path("twin_model", "pt", subdir_name=subdir_name))
        render(model_twin.report(), "model_report", exp=exp, subdir_name=subdir_name)
        loss_suite_optics = LossSuite(
            train_loss=optics_loss, train_data=data_random, eval_losses=optics_eval_losses,
            validation_data=data_validation,
        )
        I_true, I_pred = compare_prediction(model_twin, loss_suite=loss_suite_optics)
        render_image(I_true, cmap="gray", name="sample_camera_true", exp=exp, subdir_name=subdir_name)
        render_image(I_pred, cmap="gray", name="sample_camera_pred", exp=exp, subdir_name=subdir_name)

    # ---------------------------------------------------------------
    # Stage 1: camera (affine) calibration
    # ---------------------------------------------------------------
    if run_stages[0]:
        if fix_camera or ((module_flags is not None) and (module_flags.camera is not None) and (not module_flags.camera)):
            print("Camera parameters are fixed; skipping camera training.")
            loss_camera = np.array([])
        else:
            loss_suite_camera = LossSuite(
                train_loss=None, train_data=data_checker
            )
            stage = CameraEstimateCVStage("Camera", loss_suite=loss_suite_camera, param_groups=[],
                                        iterations=1, batch_size=batch_size,
                                        micro_batch_size=micro_batch_size)
            histories["camera"], _ = stage.run(model_twin, exp=exp)
    else:
        print("Skipping camera training stage.")

    # ---------------------------------------------------------------
    # Stage 2: optics training
    # ---------------------------------------------------------------
    if run_stages[1]:
        loss_suite_optics = LossSuite(
            train_loss=optics_loss, train_data=data_random, eval_losses=optics_eval_losses,
            validation_data=data_validation,
        )
        stage = OpticsStage(
            "Optics", loss_suite=loss_suite_optics, param_groups=[
                ParamGroup("optics_no_pupil", lr=5e-2),
                ParamGroup("pupil", lr=2e-1, weight_decay=0),
                ParamGroup("camera", lr=5e-3, weight_decay=0),
            ],
            iterations=it_optics, batch_size=batch_size, micro_batch_size=micro_batch_size,
            fix_camera=fix_camera, module_flags=module_flags,
        )
        histories["optics"], _ = stage.run(model_twin, exp=exp)
    else:
        print("Skipping optics training stage.")

    # ---------------------------------------------------------------
    # Stage 3: background training
    # ---------------------------------------------------------------
    if run_stages[2]:
        # Skip background training if the background flag is explicitly set to False.
        if (module_flags is not None) and (module_flags.background is not None) and (not module_flags.background):
            print("Background training skipped because background is disabled.")
        else:
            loss_suite_background = LossSuite(
                train_loss=background_loss, train_data=data_random, eval_losses=back_eval_losses,
                validation_data=data_validation,
            )
            stage = BackgroundStage(
                "Background", loss_suite=loss_suite_background,
                param_groups=[ParamGroup("background", lr=3e-2)],
                iterations=it_back, batch_size=batch_size, micro_batch_size=micro_batch_size,
                module_flags=module_flags, sat_mask=sat_mask,
            )
            histories["background"], _ = stage.run(model_twin, exp=exp)
    else:
        print("Skipping background training stage.")

    return histories