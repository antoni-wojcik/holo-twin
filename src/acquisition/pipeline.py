"""
Shared hardware acquisition pipeline: given a config and the live SLM /
camera classes to use, acquire gratings (affine calibration), a (1,1) ZOD
probe, training noise, and validation noise, all into one ExpIO run.
"""
from dataclasses import dataclass
from typing import Optional, Type

from src.io.base_config import BaseConfig
from src.io.data_io import ExpIO
from src.acquisition.data import grating_pattern, mask_pattern, random_pattern
from src.acquisition.capture import acquire_batch
from src.hardware.islm import ISLM
from src.hardware.icamera import ICamera


@dataclass
class AcquisitionConfig(BaseConfig):
    """
    Everything needed to run one hardware acquisition job: gratings at a
    range of pitches (affine calibration), a (1,1)-pitch ZOD probe, and a
    batch of random holograms (optics/background training data).
    """

    # --- experiment bookkeeping ---
    exp_base_name: str = "citl_acquisition"
    exp_name: str = "run"
    exp_description: str = ""

    # --- hardware settings ---
    wavelength: float = 633.0    # nm, laser wavelength
    slm_timeout: float = 5.0     # seconds, SLM connection timeout
    cam_gain: float = 0.0        # dB

    # --- exposure (switched mid-run: gratings/mask are bright and short, noise spreads power out) ---
    cam_exposure_gratings: float = 0.05   # seconds -- used for both the grating batch and the ZOD-mask probe
    cam_exposure_noise: float = 0.2       # seconds -- used for the random-hologram batch

    # --- pattern counts ---
    num_checkers: int = 15                    # number of distinct grating pitches to probe, see pitch_for_index()
    num_noise: int = 150                      # number of random holograms to acquire for background training
    num_validation: int = 20                  # number of random holograms to acquire for held-out validation
    check_grayscale_modulation: float = 0.4   # grating modulation depth in [0, 1]

    # --- output ---
    save_images: bool = True   # if False, still saves .npy arrays but skips .png previews for quick viewing


def run_acquisition(
    cfg: AcquisitionConfig,
    slm_cls: Type[ISLM],
    camera_cls: Type[ICamera],
    exp: Optional[ExpIO] = None,
):
    """
    Acquire, in order, into one ExpIO run:

        checkers/holos/*.npy      checkers/captures/*.npy      -- cfg.num_checkers gratings, one per pitch
        mask/holos/*.npy          mask/captures/*.npy          -- single (1,1) ZOD probe
        noise/holos/*.npy         noise/captures/*.npy         -- cfg.num_noise random holograms (training)
        validation/holos/*.npy    validation/captures/*.npy    -- cfg.num_validation random holograms (held out)
    """
    if exp is not None:
        cfg.save(exp.get_new_path("config", "json"))

    slm_shape = slm_cls.get_shape()
    grayscale_range = slm_cls.get_grayscale_range()

    with slm_cls(wavelength=cfg.wavelength, use_memory_mode=False, set_wavelength=False, timeout=cfg.slm_timeout) as slm, \
         camera_cls(exposure_time=cfg.cam_exposure_gratings, gain=cfg.cam_gain) as camera:

        acquire_batch(
            slm, camera, cfg.num_checkers,
            pattern_fn=lambda i: grating_pattern(slm_shape, i, modulation=cfg.check_grayscale_modulation, device="cpu"),
            exp=exp, subdir_name="checkers", grayscale_range=grayscale_range, save_images=cfg.save_images,
        )
        acquire_batch(
            slm, camera, 1,
            pattern_fn=lambda i: mask_pattern(slm_shape, device="cpu"),
            exp=exp, subdir_name="mask", grayscale_range=grayscale_range, save_images=cfg.save_images,
        )

        # Noise holograms spread power out over the whole far field, so they need a longer exposure
        # than the bright, concentrated gratings/mask -- switch it on the already-open camera.
        camera.exposure_time = cfg.cam_exposure_noise
        acquire_batch(
            slm, camera, cfg.num_noise,
            pattern_fn=lambda i: random_pattern(slm_shape, seed=i, device="cpu"),
            exp=exp, subdir_name="noise", grayscale_range=grayscale_range, save_images=cfg.save_images,
        )

        # Held-out validation noise: same pattern generator and exposure as the training noise
        # batch above, just continuing the seed sequence past cfg.num_noise so it never overlaps.
        acquire_batch(
            slm, camera, cfg.num_validation,
            pattern_fn=lambda i: random_pattern(slm_shape, seed=cfg.num_noise + i, device="cpu"),
            exp=exp, subdir_name="validation", grayscale_range=grayscale_range, save_images=cfg.save_images,
        )