"""
Load the twin + ground truth saved by run_training.py, run CGH against the
twin, and "capture" the result by rendering it through the ground-truth
model's own camera model -- the sim stand-in for a real hardware capture.
"""
import torch

from src.twin.model import HoloSystem, ModuleFlags
from src.cgh.targets import get_simple_target
from src.cgh.pipeline import SimCGHConfig, run_cgh, cgh_predict_images
from src.loss.losses import nmse_loss
from src.io.data_io import ExpIO, data_path

# ===============================
# CONFIG -- change this to set up a new run
# ===============================
DEFAULT_CONFIG = SimCGHConfig(
    twin_path="2026-07-08_sim_experiment_run/Background/twin_model.pt",       # last training stage's checkpoint
    ground_truth_path="2026-07-08_sim_experiment_run/ground_truth/model_true.pt",
    exp_base_name="sim_experiment", exp_name="cgh",
    exp_description="CGH on a trained sim twin, evaluated against ground truth.",
    pattern="circle",  # "circle" or "checkerboard"
    batch_size=4, micro_batch_size=2, num_epochs=200,
    module_flags=ModuleFlags(camera=False),                # twin: optimize against the far field
    ground_truth_flags=ModuleFlags(background=False),      # ground truth: focus on the deflected field, not stray light
    compute=True,
)

def run(cfg: SimCGHConfig):
    """
    Run a single CGH job with the given config, using the twin and ground
    truth models saved by run_training.py. Saves into one ExpIO run.
    """
    exp = ExpIO(base_name=cfg.exp_base_name, exp_name=cfg.exp_name, description=cfg.exp_description)

    model_twin = HoloSystem.load(data_path(cfg.twin_path))
    model_true = HoloSystem.load(data_path(cfg.ground_truth_path))
    cfg.prepare_ground_truth(model_true)

    # fov must be applied before the target is built, since target shape depends on it
    if cfg.fov != 1.0:
        model_twin.set_fov(cfg.fov)
        model_true.set_fov(cfg.fov)

    if cfg.pattern in ("circle", "checkerboard"):
        target = get_simple_target(model_twin.geometry.far_fov_samples, pattern=cfg.pattern, device=model_twin.geometry.device)
    else:
        raise ValueError(f"Unsupported pattern for sim CGH: {cfg.pattern!r}")


    def evaluate_on_ground_truth(g: torch.Tensor, batch_idx: int):
        """The "capture" step in simulation: render g through the ground-truth
        model's own camera model, standing in for a real hardware capture.
        run_cgh() itself computes the matching predicted intensity/camera
        image from model_twin, via this same cgh_predict_images() helper."""
        # In simulation there's no physical sensor, so the "camera shape" is just the far-field grid.
        _, C_true = cgh_predict_images(model_true, g, camera_shape=tuple(target.shape), exposure_time=None)
        return C_true


    run_cgh(
        cfg, model_twin, target, nmse_loss,
        exp=exp, evaluate_fn=evaluate_on_ground_truth,
    )

    # ===============================
    # CLEANUP
    # ===============================
    print("CGH complete:", exp.path)
    exp.close()


if __name__ == "__main__":
    run(DEFAULT_CONFIG)