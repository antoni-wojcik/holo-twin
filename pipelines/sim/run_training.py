"""
Simulated twin training: builds a fresh ground-truth HoloSystem, draws
training and validation data from it on the fly (nothing touches disk),
fits a twin to it using the same shared recipe as
pipelines/exp/run_training.py, and compares the fitted twin against the
known ground truth.
"""
import numpy as np

from src.io.data_io import ExpIO
from src.training.train import DataSource
from src.acquisition.data import make_sim_batch_fn
from src.twin.model import OpticsGeometry, ModuleFlags
from src.training.build_true import build_ground_truth, GroundTruthConfig
from src.training.plotting import render
from src.training.pipeline import SimTrainingConfig, run_training


# Ground truth and twin share one set of module flags (camera left at its own default) -- so
# training difficulty reflects exactly the effects actually present in the ground truth.
_MODULE_FLAGS = ModuleFlags(lut=True, pixel=True, slm_field=True, pupil=True, background=True)

DEFAULT_CONFIG = SimTrainingConfig(
    exp_base_name="sim_experiment", exp_name="twin_fit",
    exp_description="Simulated ground truth + digital twin training.",
    seed=0,

    # Small CPU-sized toy geometry -- fast to iterate on, not meant to match real hardware.
    geometry=OpticsGeometry(
        slm_pixels=(160, 200), scale=2, device="cpu",
        camera_shape=(200, 200), wavelength=60, focal_length=100,
        fov=1, camera_affine_supersample=2,
    ),

    ground_truth=GroundTruthConfig(
        crosstalk_sigma=(0.5, 0.5),
        pixel_fill=(0.95, 0.95),
        deadspace_reflectance=0.9,
        lut_scale=4.4 * np.pi,
        num_pupil_tiles=20,
        background_reflectance=0.3,
        affine_params=None,
        module_flags=_MODULE_FLAGS,
        use_crosstalk_fast_approx=False,
        use_pupil_vignetting=False,
        ideal_affine=False,
        no_tip_tilt=True,
        seed=10,
    ),

    load_twin_path=None,
    init_lut_scale=4 * np.pi,
    init_crosstalk_sigma=(0.3, 0.3),
    init_slm_field_scale=1.0,
    num_pupil_tiles=20,
    init_deadspace_reflectance=0.8,
    use_crosstalk_fast_approx=False,
    use_crosstalk_residual=True,
    use_pupil_vignetting=False,
    use_lut_depth=False,          # off, matching use_pupil_vignetting below

    run_stages=(True, True, True),
    module_flags=_MODULE_FLAGS,
    fix_camera=False,
    mask_method="checker",

    num_checker_samples=10,
    num_noise_samples=100,
    num_validation_samples=20,

    batch_size=4,
    micro_batch_size=None,
    iterations=(200, 100),
)


def _offset_batch_fn(batch_fn, offset: int):
    """
    Shift sample indices by `offset` before generating. Used to draw the
    held-out validation set from the same seeded generator as the training
    noise set while keeping the two disjoint, mirroring how
    AcquisitionConfig continues its seed sequence past num_noise for its
    validation batch.
    """
    return lambda idx: batch_fn(idx + offset)


def run(cfg: SimTrainingConfig):
    """
    Build a fresh ground-truth HoloSystem from cfg.ground_truth, draw
    training/validation data from it on the fly, and fit a twin to it.
    Takes a fully-built config rather than building one internally, so a
    caller can run this over several configs in a loop. Returns the
    resulting experiment folder's path relative to DATA_ROOT -- e.g. for
    feeding into a downstream CGH config's load_twin_path.
    """
    exp = ExpIO(cfg.exp_base_name, cfg.exp_name, cfg.exp_description)

    gt = cfg.ground_truth
    model_true = build_ground_truth(
        cfg.geometry,
        crosstalk_sigma=gt.crosstalk_sigma,
        pixel_fill=gt.pixel_fill,
        deadspace_reflectance=gt.deadspace_reflectance,
        lut_scale=gt.lut_scale,
        num_pupil_tiles=gt.num_pupil_tiles,
        background_reflectance=gt.background_reflectance,
        affine_params=gt.affine_params,
        module_flags=gt.module_flags,
        use_crosstalk_fast_approx=gt.use_crosstalk_fast_approx,
        use_pupil_vignetting=gt.use_pupil_vignetting,
        ideal_affine=gt.ideal_affine,
        no_tip_tilt=gt.no_tip_tilt,
        seed=gt.seed,
    )
    # Saved for reference alongside the fitted twin -- report_diff() below compares against it.
    model_true.save(exp.get_new_path("model_true", "pt", subdir_name="ground_truth"))
    render(model_true.report(), "model_report", exp=exp, subdir_name="ground_truth")

    data_checker = DataSource(make_sim_batch_fn(model_true, pattern="checkerboard"), total_samples=cfg.num_checker_samples)
    data_random = DataSource(make_sim_batch_fn(model_true, pattern="random"), total_samples=cfg.num_noise_samples)
    data_mask = DataSource(make_sim_batch_fn(model_true, pattern="mask"), total_samples=1)
    data_validation = DataSource(
        _offset_batch_fn(make_sim_batch_fn(model_true, pattern="random"), cfg.num_noise_samples),
        total_samples=cfg.num_validation_samples,
    )

    run_training(
        cfg, data_checker, data_random, data_mask,
        exp=exp, data_validation=data_validation, model_true=model_true,
    )

    path = exp.relative_path
    print("Training complete. Twin stored at:", path)
    exp.close()
    return path


if __name__ == "__main__":
    run(DEFAULT_CONFIG)