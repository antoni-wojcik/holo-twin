"""
Loss functions shared by digital-twin training and CGH hologram optimization.
"""
import numpy as np
import torch
from typing import Optional
from pytorch_msssim import SSIM, MS_SSIM
from src.loss.masked_ssim import MaskedSSIM, MaskedMS_SSIM
from typing import Callable


# ---------------------------------------------------------------------
# Simple intensity losses
# ---------------------------------------------------------------------

def nmse_loss(I: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Normalized MSE: each image is mean-normalized independently before comparing.
    Good for CGH where absolute brightness is arbitrary (only the *shape* of the
    pattern matters)."""
    I_norm = I / (I.mean(dim=(-2, -1), keepdim=True))
    T_norm = T / (T.mean(dim=(-2, -1), keepdim=True))
    return ((I_norm - T_norm) ** 2).mean() / (T_norm ** 2).mean()


def mse_loss(I: torch.Tensor, T: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """MSE normalized by target energy, with optional pixel mask.
    Used for twin training where I and T should already share a scale
    (I is the twin's camera-plane prediction, T is the measured/true camera image)."""
    if mask is not None:
        I = I * mask
        T = T * mask
    mse = ((I - T) ** 2).mean()
    norm = (T ** 2).mean()
    return mse / (norm + 1e-8)


def normalized_cross_correlation(I: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Negative NCC (so minimizing it maximizes correlation). Useful as a
    structure-only loss during affine/camera-only calibration, since it's
    invariant to overall gain."""
    I_mean = I.mean(dim=(-2, -1), keepdim=True)
    T_mean = T.mean(dim=(-2, -1), keepdim=True)
    I_c, T_c = I - I_mean, T - T_mean
    num = (I_c * T_c).sum(dim=(-2, -1))
    den = torch.sqrt((I_c ** 2).sum(dim=(-2, -1)) * (T_c ** 2).sum(dim=(-2, -1))) + 1e-8
    return -(num / den).mean()


# ---------------------------------------------------------------------
# SSIM / MS-SSIM factories
# ---------------------------------------------------------------------
# These need to be *built* (they carry a mask / window as state), so we
# expose factory functions rather than bare loss functions. Call once,
# reuse the returned callable across the training loop.

def build_ssim_loss(mask: torch.Tensor = None, data_range: float = 1.0) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Returns a callable loss_fn(I, T) -> scalar. Pass a boolean/float mask
    to exclude regions (e.g. the ZOD) from the SSIM computation."""
    if mask is not None:
        ssim_fn = MaskedSSIM(data_range=data_range, mask=mask)
    else:
        ssim_fn = SSIM(data_range=data_range, size_average=True, channel=1)

    def loss(I: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return 1 - ssim_fn(I.unsqueeze(1), T.unsqueeze(1))
    return loss


def _gaussian_weights(num_scales: int = 5, mu: float = 0.0, sigma: float = 1.5) -> np.ndarray:
    """
    Normalized Gaussian weights across MS-SSIM scales, peaked at `mu` with
    spread `sigma`. Used as build_ms_ssim_loss's default per-scale
    weighting (see `use_gaussian_weights` there) instead of MS_SSIM's own
    built-in default (roughly uniform), which tends to over-weight the
    coarsest scales for these images.

    Parameters
    ----------
    num_scales : int
        Number of MS-SSIM scales (default: 5).
    mu : float
        Peak position. 0.0 centers the peak at scale 1 (finest scale).
    sigma : float
        Standard deviation controlling the decay rate. Smaller values = steeper drop-off.
    """
    scales = np.arange(num_scales)
    raw_weights = np.exp(-0.5 * ((scales - mu) / sigma) ** 2)
    return raw_weights / np.sum(raw_weights)


def build_ms_ssim_loss(
    mask: torch.Tensor = None,
    data_range: float = 1.0,
    weights: Optional[list] = None,
    use_gaussian_weights: bool = True,
    gaussian_num_scales: int = 5,
    gaussian_mu: float = 0.0,
    gaussian_sigma: float = 1.5,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Returns a callable loss_fn(I, T) -> scalar. Pass a boolean/float mask
    to exclude regions (e.g. the ZOD) from the MS-SSIM computation.

    `weights`, if given, is used as-is (explicit always wins). Otherwise,
    if `use_gaussian_weights` (default True), per-scale weights are
    computed via `gaussian_weights(gaussian_num_scales, gaussian_mu,
    gaussian_sigma)`. This only changes what THIS factory passes in by
    default -- the underlying MS_SSIM / MaskedMS_SSIM classes are
    unaffected and still fall back to their own built-in weighting when
    `weights=None` is passed to them directly. Pass
    use_gaussian_weights=False here to get that library default instead.
    """
    if weights is None and use_gaussian_weights:
        weights = _gaussian_weights(num_scales=gaussian_num_scales, mu=gaussian_mu, sigma=gaussian_sigma)

    if mask is not None:
        ms_ssim_fn = MaskedMS_SSIM(data_range=data_range, mask=mask, weights=weights)
    else:
        ms_ssim_fn = MS_SSIM(data_range=data_range, size_average=True, channel=1, weights=weights)

    def loss(I: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return 1 - ms_ssim_fn(I.unsqueeze(1), T.unsqueeze(1))
    return loss


# To build this loss, call for instance "build_structure_loss(build_ssim_loss(mask), mask)"
def build_structure_loss(
        ssim_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        mask: torch.Tensor = None,
        detach_structure: bool = True
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Combined loss: intensity MSE scaled by (detached) structural
    dissimilarity. The detach on `structure` means SSIM only acts as a
    *reweighting* signal (down-weights the intensity loss when structure is
    already close), it never contributes gradients directly."""
    def loss(I: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        structure = ssim_loss_fn(I, T)
        intensity = mse_loss(I, T, mask=mask)
        if detach_structure:
            structure = structure.detach()
        return intensity * structure
    return loss