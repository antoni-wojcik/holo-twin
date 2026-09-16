"""
CGH: given a (trained) HoloSystem, a target intensity pattern, and
optimization params, find the grayscale hologram(s) that best reproduce the
target. Works on either the twin or the ground-truth model - pass whichever
HoloSystem we want to generate on.

This module is intentionally standalone (no dependency on the CGH pipeline
config): it's a generic "optimize g against this system" routine, reused
by src.cgh.pipeline.run_cgh but also usable on its own.
"""
import math
from typing import Optional, Callable

import numpy as np
import torch
from tqdm import tqdm

from src.twin.model import HoloSystem


def _warmup_cosine_lambda(num_epochs: int, warmup_frac: float) -> Callable[[int], float]:
    """
    LR multiplier as a function of epoch/step index: linear 0 -> 1 over the
    first `warmup_frac` of `num_epochs`, then cosine 1 -> 0 over the rest.
    Meant for use with torch.optim.lr_scheduler.LambdaLR.
    """
    warmup_steps = max(1, int(round(warmup_frac * num_epochs)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        remaining = max(1, num_epochs - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / remaining)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def generate_holograms(
    system: HoloSystem,
    target: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_epochs: int = 200,
    micro_batch_size: int = 2,
    batch_size: int = 4,
    lr: float = 5e-2,
    weight_decay: float = 1e-4,
    init_g: Optional[torch.Tensor] = None,
    mask_centre: bool = True,
    dark_penalty: bool = True,
    lr_warmup_frac: Optional[float] = None,
):
    """
    Optimizes a grayscale hologram `g` (shape (batch_size, *slm_shape)) such
    that the time-averaged far-field intensity produced by `system` matches
    `target` under `loss_func(I_pred, target) -> scalar`.

    `system`'s own parameters are expected to already be frozen by the
    caller (this function only ever updates `g`). Disabling the camera
    model means we are optimizing against the *far-field* intensity rather
    than the camera-plane image - typically what we want for CGH, since the
    camera model (affine warp, saturation) is an artifact of measurement,
    not of the optical system we are designing a hologram for. Disabling
    background is often useful too, since in most cases there is too much
    power in the ZOD to optimize against, and we want to focus on the
    deflected field. (Both are the caller's responsibility - see
    `CGHConfig.module_flags` in src.cgh.pipeline.)

    Parameters
    ----------
    system : HoloSystem
        The (frozen) HoloSystem to generate camera-plane images from.
    target : torch.Tensor, shape (H, W)
        The target intensity pattern to match.
    loss_fn : callable(I_pred, I_true) -> scalar
        The loss function to minimize between the predicted and target intensities.
    num_epochs : int, optional
        The number of optimization steps to take. Default is 200.
    micro_batch_size : int, optional
        The number of holograms to process before calling backward() and accumulating gradients.
    batch_size : int, optional
        The number of holograms to optimize in parallel. Default is 4.
    lr : float, optional
        The learning rate for the AdamW optimizer. Default is 5e-2. Used as the peak LR
        when `lr_warmup_frac` is set.
    weight_decay : float, optional
        The weight decay (L2 regularization) for the AdamW optimizer. Default is 1e-4.
    init_g : torch.Tensor, optional
        An optional initial guess for the hologram(s). If None, random initialization is used.
    mask_centre : bool, optional
        Whether to mask the centre of the hologram during optimization. Default is True.
    dark_penalty : bool, optional
        Whether to apply a penalty for dark regions in the target. Default is True.
    lr_warmup_frac : float, optional
        If set, use a linear-warmup-then-cosine-decay-to-zero LR schedule instead of a
        constant `lr`: LR ramps 0 -> `lr` over the first `lr_warmup_frac` of `num_epochs`,
        then cosine-decays `lr` -> 0 over the rest. Useful when optimization diverges late
        in the run.

    Returns
    -------
    g_optim      : (batch_size, *slm_shape) float — final optimized grayscale
    loss_history : np.ndarray, shape (num_epochs,)
    """

    I_T = target.clone()
    if mask_centre:
        centre_radius = 20 * system.geometry.M / system.geometry.P
        mask = _get_mask(system, I_T.shape, centre_radius=centre_radius, device=I_T.device)
        I_T = I_T * mask

    device = system.geometry.device
    slm_shape = system.geometry.slm_pixels
    g = torch.nn.Parameter(
        init_g.to(device) if init_g is not None
        else torch.rand(batch_size, *slm_shape, dtype=torch.float, device=device)
    )
    optimizer = torch.optim.AdamW([g], lr=lr, weight_decay=weight_decay)

    scheduler = None
    if lr_warmup_frac is not None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=_warmup_cosine_lambda(num_epochs, lr_warmup_frac)
        )

    if dark_penalty:
        dark_mask = (I_T < 0.05).float()

    loss_history = np.empty(num_epochs)
    bar = tqdm(range(num_epochs), desc="Hologram Generation", unit="iter")
    for epoch in bar:
        I_pred = torch.zeros((batch_size, *target.shape), dtype=torch.float, device=system.geometry.device)
        for i in range(0, batch_size, micro_batch_size):
            g_micro = g[i:i + micro_batch_size]
            I_pred[i:i + micro_batch_size] = system(g_micro)

        I_pred = I_pred.mean(dim=0)  # time-averaged intensity across the batch
        if mask_centre:
            I_pred = I_pred * mask  # Apply the same mask to the predicted intensity

        loss = loss_fn(I_pred, I_T)

        if dark_penalty:
            I_norm = I_pred / (I_pred.mean() + 1e-8)
            dark_loss = (dark_mask * I_norm.square()).sum() / dark_mask.sum()
            loss = loss + 5 * dark_loss

        loss_history[epoch] = loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_([g], max_norm=0.5)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

        current_lr = optimizer.param_groups[0]["lr"]
        bar.set_description(f"Loss: {loss.item():.4e} | lr: {current_lr:.2e}")

        with torch.no_grad():
            g.clamp_(0, 1)

    return g.detach(), loss_history


def _get_mask(system: HoloSystem, shape: tuple, centre_radius: float, device: torch.device) -> torch.Tensor:
    """
    Create a binary mask with a circular region of zeros in the center.

    Parameters
    ----------
    shape : tuple
        The shape of the mask (H, W).
    centre_radius : float
        The radius of the central circular region to be masked (set to zero).
    device : torch.device
        The device on which to create the mask.

    Returns
    -------
    mask : torch.Tensor
        A binary mask of shape `shape` with a circular region of zeros in the center.
    """
    H, W = shape
    center_y, center_x = H // 2, W // 2
    aspect_ratio = system.geometry.Mf / system.geometry.Nf
    Y, X = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    mask = torch.ones((H, W), dtype=torch.float32, device=device)
    mask[((Y - center_y) ** 2 + ((X - center_x) * aspect_ratio) ** 2) < centre_radius ** 2] = 0
    return mask