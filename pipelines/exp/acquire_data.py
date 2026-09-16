"""
Configure and run a hardware acquisition: gratings (affine calibration) +
a single (1,1) ZOD probe + random holograms for training + held-out random
holograms for validation, all into one ExpIO run:

    <date>_citl_acquisition/<run>/
        checkers/holos/*.npy      checkers/captures/*.npy
        mask/holos/*.npy          mask/captures/*.npy
        noise/holos/*.npy         noise/captures/*.npy
        validation/holos/*.npy    validation/captures/*.npy

The acquisition loop itself lives in src/acquisition/pipeline.py's
run_acquisition() (shared, reusable) -- this script just builds a config,
picks the hardware classes, and calls it.
"""
from src.acquisition.pipeline import AcquisitionConfig, run_acquisition
from src.hardware.slm_santec.slm import SLMSantec as SLM
from src.hardware.camera_thor.camera_cs126mu import CameraThorlabs as Camera
from src.io.data_io import ExpIO


DEFAULT_CONFIG = AcquisitionConfig(
    exp_base_name="training_data", 
    exp_name="run",
    exp_description="Checkerboard + ZOD-mask + random-hologram acquisition for digital twin fitting.",
    wavelength=633, 
    slm_timeout=5,
    cam_gain=0,
    cam_exposure_gratings=0.05,   # gratings/mask are bright, short exposure
    cam_exposure_noise=0.2,       # noise holograms spread power out, need longer exposure
    num_checkers=4,              # how many distinct pitches to probe with, see pitch_for_index()
    num_noise=150,                # how many random holograms to acquire for background training
    num_validation=10,            # how many held-out random holograms to acquire for validation
    check_grayscale_modulation=0.4,
    save_images=True,             # if False, still acquires + saves .npy arrays, but skips .pngs for quick viewing
)


def run(cfg: AcquisitionConfig):
    """
    Run one hardware acquisition and return the resulting experiment
    folder's path relative to DATA_ROOT -- e.g. for feeding straight into
    a downstream TrainingConfig.acquisition_path.
    """
    exp = ExpIO(base_name=cfg.exp_base_name, exp_name=cfg.exp_name, description=cfg.exp_description)

    run_acquisition(cfg, SLM, Camera, exp=exp)

    path = exp.relative_path
    print("Acquisition complete. Data stored at:", path)
    exp.close()
    return path


if __name__ == "__main__":
    run(DEFAULT_CONFIG)