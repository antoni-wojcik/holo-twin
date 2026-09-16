"""
Build an *untrained* twin HoloSystem, with reasonable-but-noisy initial
parameters, away from singularities. Used both as the starting point for
simulation experiments and experimental (real-data) training.
"""
import torch
import numpy as np
from src.twin.model import OpticsGeometry, HoloSystem, AberrCoefficients
from typing import Optional
from src.training.zernike import Zernike


def build_twin(
    geometry: OpticsGeometry,
    init_lut_scale: float = 2 * np.pi,
    init_crosstalk_sigma: tuple = (0.3, 0.3),
    init_slm_field_scale: float = 1.0,
    num_pupil_tiles: int = 40,
    init_deadspace_reflectance: float = 0.8,

    use_crosstalk_fast_approx: bool = True,
    use_crosstalk_residual: bool = True,
    use_pupil_vignetting: bool = False,
    use_lut_depth: bool = False,
) -> HoloSystem:
    """
    Build an untrained twin with a smooth, noisy initial SLM field --
    reasonable enough to train from, but away from singularities (a flat
    or zero field has degenerate/vanishing gradients in several modules).
    Everything is left active and unfrozen; callers apply their own
    active/frozen overrides afterwards (see HoloSystem.set_active /
    set_unfrozen, and TrainingConfig.module_flags).

    Parameters
    ----------
    geometry : OpticsGeometry
        Physical system description; determines the initial field's shape.
    init_lut_scale : float
        Initial total phase span of the grayscale-to-phase LUT, in radians.
    init_crosstalk_sigma : tuple
        Initial (y, x) Gaussian sigma of the pixel-crosstalk kernel, in pixels.
    init_slm_field_scale : float
        Amplitude multiplier applied to the generated smooth field BEFORE
        it's handed to HoloSystem -- baked into the initial tensor itself.
    num_pupil_tiles : int
        Number of OTF tiles per far-field dimension for the pupil module.
    init_deadspace_reflectance : float
        Initial complex deadspace reflectance, relative to unit electrode reflectance.
    use_crosstalk_fast_approx : bool
        Use the FFT-crop crosstalk approximation instead of the full CZT (faster, less exact).
    use_crosstalk_residual : bool
        Add a learnable residual on top of the Gaussian crosstalk kernel.
    use_pupil_vignetting : bool
        Apply a learnable near-field vignetting aperture in the pupil module.
    use_lut_depth : bool
        Whether the LUT has a learnable per-pixel depth map (spatially-varying phase scale).

    Returns
    -------
    HoloSystem
        Untrained twin, everything active and unfrozen.
    """
    slm_shape = geometry.slm_pixels
    init_E_in_twin = get_smooth_field(
        slm_shape, amp_gauss_sigma=1, noise_phase_mag=0.5, noise_amp_mag=0.1,
    ) * init_slm_field_scale
    init_pupil_coeffs = AberrCoefficients.from_tensor(torch.randn(AberrCoefficients.NUM_COEFFS, device=geometry.device) * 1e-3)
    system = HoloSystem(
        geometry=geometry,
        lut_n_bins=20,
        slm_field_init_field=init_E_in_twin,
        crosstalk_init_sigma=init_crosstalk_sigma,
        crosstalk_pixel_samples=3,
        crosstalk_fast_approx=use_crosstalk_fast_approx,
        crosstalk_use_residual=use_crosstalk_residual,
        pixel_init_deadspace_reflectance=init_deadspace_reflectance,
        pupil_num_tiles=num_pupil_tiles,
        pupil_init_coeffs=init_pupil_coeffs,
        lut_init_scale=init_lut_scale,
        lut_use_depth=use_lut_depth,
        pupil_use_vignetting=use_pupil_vignetting,
    )

    return system


# =============================================================================
# Utility: smooth aberration initialiser (unchanged from original)
# =============================================================================

def get_smooth_field(
    slm_pixels: tuple,
    zernike_coeffs: Optional[np.ndarray] = None,
    amp_gauss_sigma: float = 0.5,
    amp_gauss_offset: tuple = (0, 0),
    noise_phase_mag: float = 0.0,
    noise_amp_mag: float = 0.0,
    remove_ramp: bool = False,
) -> torch.Tensor:
    """
    Build a smooth complex field on the SLM plane via Zernike phase and
    Gaussian amplitude, with optional additive noise on both channels.

    Returns
    -------
    torch.Tensor  complex64, shape slm_pixels
    """
    num_zernikes = 15
    zernike = Zernike(slm_shape=slm_pixels, num_zernikes=num_zernikes)

    aberr = np.ones(slm_pixels, dtype=np.complex64)

    if remove_ramp and zernike_coeffs is not None:
        _zernike_coeffs = np.copy(zernike_coeffs)
        _zernike_coeffs[0] = 0.0 # remove piston
        _zernike_coeffs[1] = 0.0 # remove y-tilt
        _zernike_coeffs[2] = 0.0 # remove x-tilt

    else:
        _zernike_coeffs = zernike_coeffs

    # --- phase ---
    phi = zernike.get_phase(_zernike_coeffs) if _zernike_coeffs is not None \
        else np.zeros(slm_pixels, dtype=np.float32)
    if noise_phase_mag != 0:
        noise_phase = np.random.rand(*slm_pixels).astype(np.float32) * noise_phase_mag
        phi = phi + noise_phase
    aberr = aberr * np.exp(1j * phi)

    # --- amplitude ---
    if amp_gauss_sigma != 0:
        H, W = slm_pixels
        dim_max = max(H, W)
        H_norm, W_norm = H / dim_max, W / dim_max
        Y, X = np.meshgrid(
            np.linspace(-H_norm, H_norm, H),
            np.linspace(-W_norm, W_norm, W),
            indexing="ij",
        )
        amp = np.exp(
            -(
                (X - amp_gauss_offset[0]) ** 2 + (Y - amp_gauss_offset[1]) ** 2
            ) / (2 * amp_gauss_sigma ** 2)
        )
    else:
        amp = np.ones(slm_pixels, dtype=np.float32)

    if noise_amp_mag != 0:
        noise_amp = np.random.rand(*slm_pixels).astype(np.float32) * noise_amp_mag
        amp = amp + noise_amp
        amp = np.abs(amp)
        amp = amp / np.max(amp)

    aberr = aberr * amp
    return torch.tensor(aberr, dtype=torch.cfloat)