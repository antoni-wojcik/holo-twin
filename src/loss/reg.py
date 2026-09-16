"""
Regularization losses for twin training. 
"""
# ---------------------------------------------------------------------
# Regularization (twin training only — not used by CGH)
# ---------------------------------------------------------------------

import torch
import torch.nn.functional as F

from src.twin.model import HoloSystem

def laplacian(x: torch.Tensor) -> torch.Tensor:
    """Discrete 2D Laplacian, Neumann-like via replicate padding."""
    kernel = torch.tensor([[0., 1., 0.],
                            [1., -4., 1.],
                            [0., 1., 0.]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(1)
    x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
    return F.conv2d(x_pad, kernel).squeeze()


def field_smoothness_reg(system: HoloSystem, lambda_field: float = 1e-3, lambda_background: float = 0.0) -> torch.Tensor:
    """
    Smoothness regularization on the learnable complex fields.

    Parameters
    ----------
    system : HoloSystem
        The holographic system containing the SLM field and background field.
    lambda_field : float, optional
        Weight for the SLM field smoothness regularization term. Default is 1e-3.
    lambda_background : float, optional
        Weight for the background field smoothness regularization term. Default is 0.0.

    lambda_background defaults to 0 (off) since background regularization
    hasn't been validated yet.

    Returns
    -------
    torch.Tensor
        The computed smoothness regularization loss.
    """
    E = system.slm_field.field # (P, Q) complex
    lap_real, lap_imag = laplacian(E.real), laplacian(E.imag)
    reg = lambda_field * (lap_real ** 2 + lap_imag ** 2).mean()

    if lambda_background > 0:
        Eb = system.background.field
        lap_br, lap_bi = laplacian(Eb.real), laplacian(Eb.imag)
        reg = reg + lambda_background * (lap_br ** 2 + lap_bi ** 2).mean()

    return reg


def total_variation(x, eps=1e-6):
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    dx = dx[..., :-1, :]   # crop to common shape
    dy = dy[..., :, :-1]
    return torch.sqrt(dx**2 + dy**2 + eps).mean()


def total_variation_circular(phi: torch.Tensor) -> torch.Tensor:
    """
    TV on a phase field using the wrapped angular difference.
    """
    def wrap_diff(d):
        return (d + torch.pi) % (2 * torch.pi) - torch.pi
 
    dy = wrap_diff(phi[1:, :] - phi[:-1, :])
    dx = wrap_diff(phi[:, 1:] - phi[:, :-1])
    return dy.abs().mean() + dx.abs().mean()
 
def total_variation_reg(
        field: torch.Tensor, 
        lambda_tv_amp: float = 1e-3,
        lambda_tv_phase: float = 1e-3,
        lambda_l2: float = 1e-4
        ) -> torch.Tensor:
    E = field
    tv_amp = total_variation(E.abs())              # your existing amplitude TV is fine, no wraparound issue
    tv_phase = total_variation_circular(E.angle())  # replaces the naive version
    lap_real, lap_imag = laplacian(E.real), laplacian(E.imag)
    l2_small = (lap_real ** 2 + lap_imag ** 2).mean()
    return lambda_tv_amp * tv_amp + lambda_tv_phase * tv_phase + lambda_l2 * l2_small


def total_variation_reg_lut_depth(lut_depth: torch.Tensor, lambda_tv: float = 1e-3) -> torch.Tensor:
    """
    Total variation regularization on the LUT depth map.

    Parameters
    ----------
    lut_depth : torch.Tensor
        The LUT depth map (2D tensor).
    lambda_tv : float, optional
        Weight for the total variation regularization term. Default is 1e-3.

    Returns
    -------
    torch.Tensor
        The computed total variation regularization loss for the LUT depth.
    """
    tv = total_variation(lut_depth)
    return lambda_tv * tv
