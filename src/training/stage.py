"""
TrainingStage bundles the part that's identical across every training stage
(build optimizer param groups -> build scheduler -> call train_model ->
plot report/loss/activity/eval) so a stage definition in the experiment
script reduces to just the things that actually differ: which modules are
on/frozen, what data feeds it, and the LR per param group.
It's a template-method class: `run()` is fixed (setup -> train -> report/save,
identical for every stage), while `setup()` and `_train_loop()` are the two override points.

Override `setup()` for stages that just need different enable/disable/
freeze/reshape logic before training starts (the common case -- see
CameraStage/CoarseOpticsStage/FineOpticsStage/BackgroundStage below).

Override `_train_loop()` only when the training procedure itself is
structurally different -- e.g. AlternatingStage, which switches which
param group is unfrozen every few epochs instead of training everything
jointly for the whole stage. Overriding this does NOT require touching
setup(), build_optimizer_and_scheduler(), or the reporting/saving logic --
those stay shared.
"""
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict
import numpy as np
import torch
import gc
import matplotlib.pyplot as plt
from scipy.ndimage import grey_opening, gaussian_filter, label, maximum_filter
from scipy.spatial import KDTree as cKDTree
import cv2

from src.twin.model import HoloSystem, ModuleFlags
from src.io.data_io import ExpIO
from src.training.train import train_model, LossSuite
from src.training.schedules import make_scheduler
from src.training.plotting import plot_loss, plot_convergence, plot_activity, render, render_image


@dataclass
class ParamGroup:
    """
    A parameter group for the optimizer, specifying which model parameters to optimize,
    the learning rate, weight decay, and LR schedule.

    Parameters
    ----------
    params : str
        Key into PARAM_ACCESSORS to select which model parameters this group optimizes.
        Choose from: "all", "optics", "optics_no_pupil", "pupil", "camera", "background", "optics_no_pupil_lut_depth", "lut_depth".
        "optics" includes SLM field, LUT, pixel, and pupil modules; "optics_no_pupil" excludes the pupil module.
    lr : float
        Learning rate for this param group.
    weight_decay : float, optional
        Weight decay for this param group. Default is 1e-4.
    schedule : tuple, optional
        Tuple of (warmup_frac, decay_end_frac, floor) for the LR schedule.
        Default is (0.5, 1.0, 0.0) for a standard warmup + cosine decay to zero.
    """
    params: str
    lr: float
    weight_decay: float = 1e-4
    schedule: tuple = (0.5, 1.0, 0.0)  # (warmup_frac, decay_end_frac, floor)

def _optics_no_pupil_lut_depth_gen(m: HoloSystem):
    yield from m.slm_field.parameters()
    yield from m.pixel.parameters()
    yield m.lut.phase_slopes_raw

def _lut_depth_gen(m: HoloSystem):
    yield m.lut.depth_raw

def _camera_no_shift_gen(m: HoloSystem):
    yield m.camera.rotation_raw
    yield m.camera.scale_raw

PARAM_ACCESSORS: Dict[str, Callable[[HoloSystem], List[torch.nn.Parameter]]] = {
    "all": lambda m: m.parameters(),
    "optics": lambda m: m.params(default=True, flags=ModuleFlags(camera=False, background=False)),
    "optics_no_pupil": lambda m: m.params(default=True, flags=ModuleFlags(camera=False, background=False, pupil=False)),
    "optics_no_pupil_lut_depth": _optics_no_pupil_lut_depth_gen,
    "lut_depth": _lut_depth_gen,
    "pupil": lambda m: m.pupil.parameters(),
    "camera": lambda m: m.camera.parameters(),
    "camera_no_shift": _camera_no_shift_gen,
    "background": lambda m: m.background.parameters(),
}

def compare_prediction(model: HoloSystem, loss_suite: LossSuite):
    """Compare the model's prediction to the true camera image for a single batch.
    Returns the true image, predicted image, and labels for plotting."""
    g, I_true = loss_suite.train_data.batch_fn(torch.arange(1))
    with torch.no_grad():
        I_pred = model(g)
    return I_true[0], I_pred[0]

class TrainingStage:
    """
    Base class for one stage of twin training.

    Subclass and override `setup(self, model)` for stages that only differ
    in which modules are enabled/frozen/reshaped beforehand (most stages).
    Override `_train_loop(self, model)` for stages whose training procedure
    is fundamentally different (rare -- see AlternatingStage).

    Parameters
    ----------
    name : str
        Name of the stage, used for logging and saving.
    loss_suite : LossSuite
        LossSuite containing the training loss, training data, and optional evaluation losses for this stage.
    param_groups : list of ParamGroup
        List of optimizer param groups for this stage.
    iterations : int
        Number of epochs to train for this stage.
    batch_size : int
        Batch size for this stage.
    micro_batch_size : int, optional
        Micro-batch size for gradient accumulation. If None, defaults to batch_size.
    grad_clip_norm : float, optional
        Gradient clipping norm for this stage. Default is 1.0.
    show_activity : bool, optional
        Whether to plot parameter activity after training. Default is True.
    save_fields : bool, optional
        Whether to save SLM and background fields after training. Default is True.
    module_flags : ModuleFlags, optional
        Optional flags to override which modules are active/unfrozen for this stage.

    Example Usage
    -------------
    stage = CameraStage("Camera", loss_suite=loss_suite, param_groups=[ParamGroup("all", lr=1e-1)],
                         iterations=50, batch_size=4)
    history, activity_log = stage.run(model_twin, exp=exp)
    """

    def __init__(
        self,
        name: str,
        loss_suite: LossSuite,
        param_groups: List[ParamGroup],
        iterations: int,
        batch_size: int,
        micro_batch_size: Optional[int] = None,
        grad_clip_norm: float = 1.0,
        show_activity: bool = True,
        save_fields: bool = True,
        module_flags: Optional[ModuleFlags] = None,
    ):
        self.name = name.lower().replace(" ", "_") + "_stage"
        self.loss_suite = loss_suite
        self.param_groups = param_groups
        self.iterations = iterations
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size or batch_size
        self.grad_clip_norm = grad_clip_norm
        self.show_activity = show_activity
        self.save_fields = save_fields
        self.module_flags = module_flags

    # ------------------------------------------------------------------
    # Override points
    # ------------------------------------------------------------------

    def setup(self, model: HoloSystem):
        """Enable/disable/freeze/unfreeze/reshape the model before this
        stage starts. No-op by default."""
        pass

    def build_optimizer_and_scheduler(self, model: HoloSystem, param_groups: List[ParamGroup] = None, iterations: int = None):
        """Broken out so _train_loop overrides (e.g. AlternatingStage) can
        build a fresh optimizer/scheduler per phase without duplicating
        this logic."""
        param_groups = param_groups if param_groups is not None else self.param_groups
        iterations = iterations if iterations is not None else self.iterations

        optimizer = torch.optim.AdamW([
            {
                "params": list(PARAM_ACCESSORS[g.params](model)),
                "lr": g.lr,
                "weight_decay": g.weight_decay,
            }
            for g in param_groups
        ])
        scheduler = make_scheduler(optimizer, iterations, [g.schedule for g in param_groups])
        return optimizer, scheduler

    def _train_loop(self, model: HoloSystem):
        """Default: a single train_model() call for self.iterations epochs.
        Override for a structurally different training procedure."""
        optimizer, scheduler = self.build_optimizer_and_scheduler(model)
        return train_model(
            model, optimizer, self.loss_suite, num_epochs=self.iterations,
            micro_batch_size=self.micro_batch_size, batch_size=self.batch_size,
            scheduler=scheduler, grad_clip_norm=self.grad_clip_norm, name=self.name,
        )

    # ------------------------------------------------------------------
    # Template method -- do not override
    # ------------------------------------------------------------------

    def run(self, model: HoloSystem, exp: ExpIO = None):
        """
        Returns
        -------
        history : dict
            {name: np.ndarray of shape (iterations,)} exactly as returned by
            train_model() -- the training objective, the eval specs, and (if
            the loss suite had validation_data) the per-epoch validation
            mean/sem/std/min/max. Already rendered and saved by _report()
            below; returned as well so a caller that wants to plot it
            itself (e.g. a figure script laying several runs out together)
            does not have to re-run training or re-read it from disk.
        activity_log : dict
            {parameter name: list of per-epoch mean absolute gradients}.
        """
        self.setup(model)

        # Override module flags if provided (e.g. to disable LUT for a stage)
        self._apply_flags(model)

        print(f"TRAINING {self.name.upper()}")
        history, activity_log = self._train_loop(model)
        self._report(model, history, activity_log, exp)
        print(f"{self.name.upper()} TRAINING COMPLETE")

        self._cleanup()  # free memory
        return history, activity_log

    def _apply_flags(self, model: HoloSystem):
        if self.module_flags is not None:
            # Override only the flags that are not None; leave the rest as-is
            model.set_active(default=None, flags=self.module_flags)
            model.set_unfrozen(default=None, flags=self.module_flags)

    def _cleanup(self):
        """Free memory after this stage is done. Call after run() if you want to
        free GPU memory before the next stage."""
        del self.loss_suite, self.param_groups

        # detect if torch is using the cuda backend and free memory if so
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def _report(self, model: HoloSystem, history: dict, activity_log: dict, exp: ExpIO):
        render(model.report(), "model_report", exp=exp, subdir_name=self.name)
        # `history` is already {name: array}, self-describing, exactly as returned by
        # train_model() (or, for CameraEstimateCVStage, its own fit loop) -- no conversion
        # step needed to plot or save it. Stages that track a "structure" eval spec (optics,
        # background) get the richer convergence plot (train w/ + w/o reg, validation mean
        # +/- SEM); stages without one (camera; or any stage run with skip_eval_losses=True)
        # fall back to the plain per-key loss plot. No combined end-of-run plot -- each stage
        # renders its own here as it finishes.
        has_data = bool(history) and any(len(v) > 0 for v in history.values())
        if has_data:
            if "eval: structure" in history:
                render(plot_convergence(history), "convergence", exp=exp, subdir_name=self.name)
            else:
                render(plot_loss(history), "loss_history", exp=exp, subdir_name=self.name)
        if self.show_activity and activity_log != {}:
            render(plot_activity(activity_log, f"{self.name} Parameter Activity"), "activity_log", exp=exp, subdir_name=self.name)
        if exp is not None:
            model.save(exp.get_new_path("twin_model", "pt", subdir_name=self.name))
            # plot prediction vs true camera image
            I_true, I_pred = compare_prediction(model, loss_suite=self.loss_suite)
            render_image(I_true, cmap="gray", name="sample_camera_true", exp=exp, subdir_name=self.name)
            render_image(I_pred, cmap="gray", name="sample_camera_pred", exp=exp, subdir_name=self.name)
            if has_data:
                exp.save_npy(history, "loss_history", subdir_name=self.name)
            if activity_log != {}:
                exp.save_npy(activity_log, "activity_log", subdir_name=self.name)
            if self.save_fields:
                if model.slm_field.is_active:
                    E_slm = model.slm_field.field.detach()
                    render_image(E_slm.abs(), cmap="hot", name="field_slm_abs", exp=exp, subdir_name=self.name)
                    render_image(E_slm.angle(), cmap="twilight", name="field_slm_phase", exp=exp, subdir_name=self.name)
                if model.background.is_active:
                    E_back, U_back = model.background.get_filtered(mode="both")
                    render_image(E_back.abs(), cmap="hot", name="background_slm_abs", exp=exp, subdir_name=self.name)
                    render_image(E_back.angle(), cmap="twilight", name="background_slm_phase", exp=exp, subdir_name=self.name)
                    render_image(U_back.abs(), cmap="hot", name="background_far_abs", exp=exp, subdir_name=self.name)
                    render_image(U_back.angle(), cmap="twilight", name="background_far_phase", exp=exp, subdir_name=self.name)
    
# ---------------------------------------------------------------------
# Concrete stages -- each just overrides setup(); everything else shared
# ---------------------------------------------------------------------

class CameraStage(TrainingStage):
    def setup(self, model):
        # Disable optics modules, enable camera, freeze optics
        model.set_active(default=False, flags=ModuleFlags(camera=True))
        model.set_unfrozen(default=False, flags=ModuleFlags(camera=True))


class OpticsStage(TrainingStage):
    def __init__(self, *args, fix_camera: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fix_camera = fix_camera

    def setup(self, model):
        # Enable optics, disable and freeze background, optionally freeze camera
        model.set_active(default=True, flags=ModuleFlags(background=False))
        model.set_unfrozen(default=True, flags=ModuleFlags(background=False,
                                                           camera=not self.fix_camera))

class BackgroundStage(TrainingStage):
    # add mask param into the init
    def __init__(self, *args, sat_mask: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.sat_mask = sat_mask

    def setup(self, model):
        # Enable all modules, but freeze optics and camera, so only background is trained
        model.set_active(default=True)
        model.set_unfrozen(default=False, flags=ModuleFlags(background=True))
        
        # clean up the SLM field to remove any remaining noise in the aperture region
        model.slm_field.clean_aperture()

    def _train_loop(self, model: HoloSystem):
        history, activity_log = super()._train_loop(model)

        mask_cam = 1 - self.sat_mask # invert so that the saturated regions are 1 and rest is 0
        mask_far = model.camera.affine_inverse(mask_cam.unsqueeze(0), square=False).squeeze(0)
        mask_far = mask_far > 0.5  # threshold to binary mask
        model.background.apply_saturation_mask(mask_far)

        return history, activity_log

# -----------------------------------------------------------------------
# Custom stages with different training procedures (override _train_loop)
# -----------------------------------------------------------------------

class CameraEstimateCVStage(TrainingStage):
    """
    Estimate the camera affine transform from grating captures with
    cv2.estimateAffine2D instead of gradient descent.

    Peak detection is DIFFERENT for the two image types, because they have
    fundamentally different pixel statistics:

    - Ideal far-field (model.propagate output): noise-free, delta-function-
      like diffraction orders often only 1-2px wide. A morphological
      opening -- essential for the measured side -- erases these entirely.
      Detected via plain local-maxima + an exclusion-zone-aware threshold
      (so the very bright zero order doesn't set the bar too high for
      dimmer outer orders).

    - Measured camera capture: has thin (~1-2px) blooming streaks running
      through every order plus sensor noise. A morphological opening
      suppresses the streaks while preserving the wider real PSF blobs;
      an added elongation filter then rejects any streak fragments that
      survive the opening as separate long/thin blobs (rather than
      compact, roughly-round diffraction spots).

    Correspondence: camera peaks are projected into far-field-normalised
    space using the model's *current* affine guess and matched to detected
    far-field peaks by mutual nearest-neighbour within a tolerance. Peaks/
    matching are cached once per sample; the match -> fit -> re-match loop
    (`refine_iters`) then only re-runs cheap geometry, tightening as the
    estimate improves.
    """

    def __init__(self, *args,
                 opening_size: int = 5,
                 blur_sigma: float = 1.0,
                 peak_threshold_rel: float = 0.15,
                 exclude_radius_px: float = 40.0,
                 min_blob_pixels: int = 3,
                 max_elongation: float = 3.0,
                 far_min_distance: int = 15,
                 far_blur_sigma: float = 0.7,
                 far_threshold_rel: float = 0.05,
                 min_peaks_per_image: int = 6,
                 match_tol: float = 0.08,
                 ransac_thresh: float = 0.02,
                 min_matches_per_image: int = 3,
                 refine_iters: int = 3,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.opening_size = opening_size
        self.blur_sigma = blur_sigma
        self.peak_threshold_rel = peak_threshold_rel
        self.exclude_radius_px = exclude_radius_px
        self.min_blob_pixels = min_blob_pixels
        self.max_elongation = max_elongation
        self.far_min_distance = far_min_distance
        self.far_blur_sigma = far_blur_sigma
        self.far_threshold_rel = far_threshold_rel
        self.min_peaks_per_image = min_peaks_per_image
        self.match_tol = match_tol
        self.ransac_thresh = ransac_thresh
        self.min_matches_per_image = min_matches_per_image
        self.refine_iters = refine_iters
        self._debug = []

    def setup(self, model):
        # Enable optics, disable and freeze background, optionally freeze camera
        model.set_active(default=True, flags=ModuleFlags(background=False))

    # ------------------------------------------------------------------
    # Far-field (synthetic, noise-free) peak detection: plain local maxima,
    # no morphological opening -- peaks here can be only 1-2px wide and an
    # opening would erase them.
    # ------------------------------------------------------------------
    def _find_peaks_far(self, img: np.ndarray, center_xy: np.ndarray) -> np.ndarray:
        img_f = img.astype(np.float64)
        if img_f.max() <= 0:
            return np.zeros((0, 2))

        smoothed = gaussian_filter(img_f, sigma=self.far_blur_sigma)

        yy, xx = np.indices(smoothed.shape)
        r2 = (xx - center_xy[0]) ** 2 + (yy - center_xy[1]) ** 2
        outside = r2 > self.exclude_radius_px ** 2

        # threshold relative to the brightest peak OUTSIDE the exclusion
        # zone, not the (usually much brighter) zero order -- otherwise
        # the outer orders never clear the bar.
        if not np.any(outside) or smoothed[outside].max() <= 0:
            return np.zeros((0, 2))
        thresh = smoothed[outside].max() * self.far_threshold_rel

        mx = maximum_filter(smoothed, size=self.far_min_distance)
        peak_mask = (smoothed == mx) & (smoothed > thresh) & outside
        ys, xs = np.nonzero(peak_mask)

        pts = []
        half = max(2, self.far_min_distance // 4)
        H, W = smoothed.shape
        for x, y in zip(xs, ys):
            y0, y1 = max(0, y - half), min(H, y + half + 1)
            x0, x1 = max(0, x - half), min(W, x + half + 1)
            patch = smoothed[y0:y1, x0:x1]
            patch = patch - patch.min()
            if patch.sum() <= 0:
                pts.append((float(x), float(y)))
                continue
            yy_p, xx_p = np.mgrid[y0:y1, x0:x1]
            pts.append(((xx_p * patch).sum() / patch.sum(),
                        (yy_p * patch).sum() / patch.sum()))
        return np.array(pts, dtype=np.float64)

    # ------------------------------------------------------------------
    # Camera (measured) peak detection: morphological opening to suppress
    # blooming streaks, plus an elongation filter to reject any streak
    # fragments that survive as separate long/thin blobs.
    # ------------------------------------------------------------------
    def _find_peaks_cam(self, img: np.ndarray, center_xy: np.ndarray) -> np.ndarray:
        img_f = img.astype(np.float64)
        if img_f.max() <= 0:
            return np.zeros((0, 2))

        opened = grey_opening(img_f, size=self.opening_size)
        smoothed = gaussian_filter(opened, sigma=self.blur_sigma)

        yy, xx = np.indices(smoothed.shape)
        r2 = (xx - center_xy[0]) ** 2 + (yy - center_xy[1]) ** 2
        outside = r2 > self.exclude_radius_px ** 2

        if not np.any(outside) or smoothed[outside].max() <= 0:
            return np.zeros((0, 2))
        thresh = smoothed[outside].max() * self.peak_threshold_rel
        mask = (smoothed > thresh) & outside

        labeled, n = label(mask)
        pts = []
        for lbl in range(1, n + 1):
            ys, xs = np.nonzero(labeled == lbl)
            if len(xs) < self.min_blob_pixels:
                continue

            # reject elongated blobs (streak fragments): compare the two
            # eigenvalues of the pixel-coordinate covariance -- a round
            # diffraction spot has a low ratio, a thin streak has a high one.
            if len(xs) >= 4:
                coords = np.stack([xs, ys], axis=1).astype(np.float64)
                coords -= coords.mean(axis=0)
                cov = np.cov(coords.T)
                eigvals = np.linalg.eigvalsh(cov)
                eigvals = np.clip(eigvals, 1e-6, None)
                elongation = np.sqrt(eigvals[1] / eigvals[0])
                if elongation > self.max_elongation:
                    continue

            w = smoothed[ys, xs]
            pts.append((float(np.sum(xs * w) / np.sum(w)),
                        float(np.sum(ys * w) / np.sum(w))))
        return np.array(pts, dtype=np.float64)

    # ------------------------------------------------------------------
    # Correspondence via projection through the current affine guess +
    # mutual nearest-neighbour, in normalised coordinates.
    # ------------------------------------------------------------------
    def _match_by_projection(self, pts_far_norm, pts_cam_norm, matrix_2x3, tol):
        if len(pts_far_norm) == 0 or len(pts_cam_norm) == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))

        pred_far = pts_cam_norm @ matrix_2x3[:, :2].T + matrix_2x3[:, 2]

        tree_far = cKDTree(pts_far_norm)
        d_cf, idx_cf = tree_far.query(pred_far)
        tree_pred = cKDTree(pred_far)
        _, idx_fc = tree_pred.query(pts_far_norm)

        src, dst = [], []
        for ci, (d, fi) in enumerate(zip(d_cf, idx_cf)):
            if d <= tol and idx_fc[fi] == ci:
                src.append(pts_cam_norm[ci])
                dst.append(pts_far_norm[fi])
        return np.array(src), np.array(dst)

    # ------------------------------------------------------------------
    def _train_loop(self, model: HoloSystem):
        g = model.geometry
        Mf, Nf = g.far_fov_samples
        Hc, Wc = g.camera_shape
        far_center = np.array([Nf / 2.0, Mf / 2.0])
        cam_center = np.array([Wc / 2.0, Hc / 2.0])

        cache = []
        self._debug = []
        with torch.no_grad():
            for i in range(self.loss_suite.train_data.total_samples):
                g_holo, I_meas = self.loss_suite.train_data.batch_fn(torch.tensor([i]))
                U = model.propagate(g_holo)
                I_ideal = model.get_intensity(U)[0].cpu().numpy()
                I_meas_np = I_meas[0].cpu().numpy()

                pk_far = self._find_peaks_far(I_ideal, far_center)
                pk_cam = self._find_peaks_cam(I_meas_np, cam_center)

                if len(pk_far) < self.min_peaks_per_image or len(pk_cam) < self.min_peaks_per_image:
                    print(f"[{self.name}] sample {i}: too few peaks "
                          f"(far={len(pk_far)}, cam={len(pk_cam)}) -- skipping")
                    continue

                far_norm = model.camera.far_pixel_to_norm(
                    torch.from_numpy(pk_far).float().to(g.device)).cpu().numpy()
                cam_norm = model.camera.camera_pixel_to_norm(
                    torch.from_numpy(pk_cam).float().to(g.device)).cpu().numpy()

                cache.append(dict(idx=i, far_norm=far_norm, cam_norm=cam_norm,
                                   I_ideal=I_ideal, I_meas=I_meas_np,
                                   pk_far=pk_far, pk_cam=pk_cam))

        if not cache:
            raise RuntimeError(
                "CameraEstimateCVStage: no sample produced enough peaks at all. "
                "Inspect stage._debug / _diagnostic_plot(); check far_threshold_rel/"
                "far_min_distance for the far-field side and peak_threshold_rel/"
                "opening_size/max_elongation for the camera side."
            )

        current_matrix = model.camera.matrix.detach().cpu().numpy()[:2]
        tol = self.match_tol
        A, err = None, None

        for it in range(self.refine_iters):
            cam_all, far_all = [], []
            self._debug = []
            for entry in cache:
                src, dst = self._match_by_projection(
                    entry["far_norm"], entry["cam_norm"], current_matrix, tol)
                self._debug.append(dict(entry, matched_cam=src, matched_far=dst))
                if len(src) >= self.min_matches_per_image:
                    cam_all.append(src)
                    far_all.append(dst)

            if not cam_all:
                raise RuntimeError(
                    f"CameraEstimateCVStage: refine iter {it}: no sample reached "
                    f"min_matches_per_image ({self.min_matches_per_image}) at "
                    f"tol={tol:.4f}. Inspect stage._debug / _diagnostic_plot() "
                    "and increase match_tol, or check the physical base scale."
                )

            cam_all = np.concatenate(cam_all).astype(np.float32)
            far_all = np.concatenate(far_all).astype(np.float32)

            A, inliers = cv2.estimateAffine2D(
                cam_all, far_all, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_thresh)
            if A is None:
                raise RuntimeError(f"cv2.estimateAffine2D failed to converge at refine iter {it}.")

            pred = cam_all @ A[:, :2].T + A[:, 2]
            err = np.linalg.norm(pred - far_all, axis=1)
            n_in = int(inliers.sum()) if inliers is not None else len(cam_all)
            print(f"[{self.name}] refine {it}: {len(cam_all)} correspondences, "
                  f"{n_in} inliers, mean err={err.mean():.5f}, tol={tol:.4f}")

            current_matrix = A
            tol = max(self.ransac_thresh * 3, tol * 0.5)

        params = model.camera.set_affine_from_correspondence(A)
        print(f"[{self.name}] FIT: rot={np.degrees(params['rotation']):.3f}°  "
              f"scale=({params['scale'][0]:.4f},{params['scale'][1]:.4f})  "
              f"shift=({params['shift'][0]:.2f},{params['shift'][1]:.2f})px")

        # Not a per-epoch training curve (this stage fits via cv2, not gradient descent) --
        # a single dict entry of sorted correspondence residuals, still fitting the same
        # {name: array} shape _report() in TrainingStage expects, so it plots/saves via the
        # plain plot_loss() fallback (no "eval: structure" key here, so plot_convergence
        # is never selected for this stage).
        return {"correspondence error (sorted)": np.sort(err)}, {}

    # ------------------------------------------------------------------
    def _diagnostic_plot(self, model=None):
        if not self._debug:
            return None
        n = len(self._debug)
        fig, axs = plt.subplots(n, 2, figsize=(8, 4 * n), squeeze=False)
        for row, d in enumerate(self._debug):
            for col, (img, pk, matched, title) in enumerate([
                (d["I_ideal"], d["pk_far"], d["matched_far"], f"sample {d['idx']}: ideal far-field"),
                (d["I_meas"], d["pk_cam"], d["matched_cam"], f"sample {d['idx']}: measured camera"),
            ]):
                ax = axs[row, col]
                ax.imshow(np.sqrt(np.clip(img, 0, None)), cmap="gray")
                if len(pk):
                    ax.scatter(pk[:, 0], pk[:, 1], s=15, facecolors="none", edgecolors="cyan", label="detected")
                if len(matched):
                    ax.scatter(matched[:, 0], matched[:, 1], s=8, c="lime", label="matched")
                ax.set_title(title, fontsize=9)
                ax.legend(fontsize=6, loc="upper right")
        fig.tight_layout()
        return fig