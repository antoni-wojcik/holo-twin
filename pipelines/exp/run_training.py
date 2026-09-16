"""
Real-hardware twin training: fits a HoloSystem to (hologram, camera image)
pairs already acquired to disk by pipelines/exp/acquire_data.py, using the
shared recipe in src/training/pipeline.py.
"""
import os
import numpy as np

from src.io.data_io import ExpIO, data_path
from src.training.train import DataSource
from src.acquisition.data import make_experimental_batch_fn
from src.training.pipeline import ExpTrainingConfig, run_training
from src.twin.model import OpticsGeometry

DEFAULT_CONFIG = ExpTrainingConfig(
    exp_base_name="twin_fit",
    exp_name=r"run",
    acquisition_path=r"paper_training_data\ap_4pi",   # relative to DATA_ROOT, written by pipelines/exp/acquire_data.py

    # --- experiment bookkeeping ---"
    exp_description = "Digital twin fit to experimental captured data.",
    seed = 0,   # torch + numpy seed, set before building/loading the twin -- for repeatable training runs

    # --- physical geometry the twin is built/trained at ---
    geometry = OpticsGeometry(
        slm_pixels=(1200, 1920), scale=2, camera_shape=(2000, 2000), camera_exposure_time=0.2, device="cuda",
    ),

    # --- twin initialisation: load an existing checkpoint, or build fresh with these flags ---
    load_twin_path = None,   # relative to DATA_ROOT; if set, build_twin() below is skipped entirely
    init_lut_scale = 4 * np.pi,
    init_crosstalk_sigma = (0.3, 0.3),
    init_slm_field_scale = 2.5,
    num_pupil_tiles = 40,
    init_deadspace_reflectance = 0.8,
    use_crosstalk_fast_approx = False,
    use_crosstalk_residual = True,

    # --- what to train ---
    run_stages = (True, True, True),        # (camera, optics, background) -- False skips a stage entirely
    module_flags = None,     # active/frozen overrides applied on top of each stage's own defaults
    fix_camera = False,                       # freeze the camera's affine params during the optics stage
    mask_method = "checker",                  # "checker" or "median", see run_training()

    # --- data: how many samples to actually train/validate on, independent of how many exist on disk ---
    num_checker_samples = 4,
    num_noise_samples = 150,
    num_validation_samples = 10,
    grayscale_range = 1024,    # quantization levels stored holograms were saved at -- must match the acquisition
                                   # that produced them (a real SLM's native range for ExpTrainingConfig, or
                                   # SimAcquisitionConfig.grayscale_range for SimStoredTrainingConfig)
    iterations = (300, 100),
    batch_size = 4,
    micro_batch_size = 2,
)

def run(cfg: ExpTrainingConfig):
    """
    Load the acquisition at cfg.acquisition_path and train a twin against
    it. Takes a fully-built config rather than building one internally, so
    a caller can run this over several configs in a loop (e.g. sweeping
    acquisition_path or a hyperparameter) without touching this file.
    Returns the resulting experiment folder's path relative to DATA_ROOT --
    e.g. for feeding into a downstream CGH config's load_twin_path.
    """
    exp = ExpIO(cfg.exp_base_name, cfg.exp_name, cfg.exp_description)

    root = data_path(cfg.acquisition_path)
    camera_shape = cfg.geometry.camera_shape
    device = cfg.geometry.device

    def loader(subdir_name: str) -> DataSource:
        batch_fn = make_experimental_batch_fn(
            os.path.join(root, subdir_name), camera_shape, device, cfg.grayscale_range
        )
        return batch_fn

    data_checker = DataSource(loader("checkers"), total_samples=cfg.num_checker_samples)
    data_random = DataSource(loader("noise"), total_samples=cfg.num_noise_samples)
    data_mask = DataSource(loader("mask"), total_samples=1)
    data_validation = DataSource(loader("validation"), total_samples=cfg.num_validation_samples)

    run_training(
        cfg, data_checker, data_random, data_mask,
        exp=exp, data_validation=data_validation,
    )

    path = exp.relative_path
    print("Training complete. Twin stored at:", path)
    exp.close()
    return path


if __name__ == "__main__":
    run(DEFAULT_CONFIG)