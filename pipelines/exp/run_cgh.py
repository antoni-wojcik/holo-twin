"""
Run CGH against the trained experimental twin, display the result on real
hardware, and capture it. Set cfg.compute=False + cfg.holo_path to
redisplay a previously-generated hologram (e.g. at a different exposure)
without re-running the optimisation.

src/cgh/pipeline.py's run_cgh() owns the whole cfg.batch_num loop --
per-batch saving into nested "<exp_name>/batch_<i>" ExpIO runs, the
holo_path/batch_i redirect when redisplaying, and averaging the batches
together at the end -- so this script just builds a config, loads the
model + target, and calls it once.
"""
import torch

from src.twin.model import HoloSystem, ModuleFlags
from src.cgh.targets import load_target_from_image, get_target_circle, get_target_checkerboard
from src.cgh.targets_meta import get_pattern
from src.cgh.pipeline import HardwareCGHConfig, run_cgh
from src.loss.losses import nmse_loss
from src.acquisition.capture import display_and_average
from src.hardware.slm_santec.slm import SLMSantec as SLM
from src.hardware.camera_thor.camera_cs126mu import CameraThorlabs as Camera
from src.io.data_io import ExpIO, data_path

# ===============================
# CONFIG -- change this to set up a new run
# ===============================
DEFAULT_CONFIG = HardwareCGHConfig(
    twin_path=r"paper_twin_fit\study_04_4pi_ap_fit\ap_4pi\background_stage\twin_model.pt",
    exp_base_name="twin_cgh",
    exp_name="run",  # nested under exp_base_name, collision-suffixed if needed
    exp_description="CGH on a trained experimental twin, displayed on real hardware.",
    pattern="custom",
    target_path=r"targets\complex.png",  # relative to DATA_ROOT
    batch_num=1, batch_size=4, micro_batch_size=1, num_epochs=300,
    fov=1.0,
    exposure_time=0.4,
    wavelength=633,
    cam_gain=0,
    pixel_fast_approx=False,
    background_aperture_mask_path=None,
    module_flags=ModuleFlags(camera=False, background=False),  # optimize against the far field, not the camera plane and discard background
    compute=True,          # False -> skip optimization, load holo_path instead
    holo_path=r"2026-08-12_twin_cgh\run",  # relative to DATA_ROOT; only used when compute=False
    mask_centre=False,
)


def _build_model_and_target(cfg: HardwareCGHConfig):
    model_twin = HoloSystem.load(data_path(cfg.twin_path))
    # fov and square_far_field must both be applied before the target is
    # built, since target shape depends on both -- order between the two
    # doesn't matter, each only replaces its own geometry field.
    if cfg.square_far_field is not None:
        model_twin.square_far_field(cfg.square_far_field)
    if cfg.fov != 1.0:
        model_twin.set_fov(cfg.fov)

    if cfg.pattern == "custom":
        target = load_target_from_image(
            data_path(cfg.target_path), shape=model_twin.geometry.far_fov_samples, device=model_twin.geometry.device
        )
    elif cfg.pattern == "circle":
        target = get_target_circle(
            shape=model_twin.geometry.far_fov_samples, device=model_twin.geometry.device
        )
    elif cfg.pattern == "checkerboard":
        target = get_target_checkerboard(
            shape=model_twin.geometry.far_fov_samples, pitch=cfg.checkerboard_pitch, device=model_twin.geometry.device
        )
    elif cfg.pattern == "meta":
        max_dim = max(model_twin.geometry.far_fov_samples)
        cell_size = 15
        num_cells = max_dim // cell_size
        pillar_size = (2, 14)
        target = get_pattern(
            shape=model_twin.geometry.far_fov_samples,
            num_cells=num_cells,
            cell_size=cell_size,
            pillar_size=pillar_size,
            device=model_twin.geometry.device
        )
    else:
        raise ValueError(f"Unsupported pattern for hardware CGH: {cfg.pattern!r}")

    return model_twin, target


def _evaluate_on_hardware(cfg: HardwareCGHConfig, g: torch.Tensor, batch_idx: int, exp: ExpIO) -> torch.Tensor:
    """The "capture" step: display g on the real SLM, average N camera
    frames, and return the resulting image. run_cgh itself computes the
    matching predicted intensity/camera image from model_twin. `exp` is
    this batch's own nested ExpIO run, passed straight through from
    run_cgh()."""
    grayscale_range = SLM.get_grayscale_range()
    with SLM(wavelength=cfg.wavelength, use_memory_mode=False, set_wavelength=False) as slm, \
        Camera(exposure_time=cfg.effective_exposure_time, gain=cfg.cam_gain) as camera:
        return display_and_average(slm, camera, g, grayscale_range, exp=exp)


def run(cfg: HardwareCGHConfig) -> str:
    """Build the model + target and hand everything to run_cgh(), which
    runs cfg.batch_num batches and averages them."""
    exp = ExpIO(base_name=cfg.exp_base_name, exp_name=cfg.exp_name, description=cfg.exp_description)
    model_twin, target = _build_model_and_target(cfg)

    run_cgh(
        cfg, model_twin, target, nmse_loss, exp=exp,
        evaluate_fn=lambda g, batch_idx, batch_exp: _evaluate_on_hardware(cfg, g, batch_idx, batch_exp),
    )

    path = exp.relative_path
    print("CGH complete. Data stored at:", path)
    exp.close()
    return path


if __name__ == "__main__":
    run(DEFAULT_CONFIG)