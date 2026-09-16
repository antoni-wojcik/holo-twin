"""
Masks used to exclude the ZOD / background region from loss computation
during optics/camera training.
"""
import torch
import numpy as np
import cv2
from src.acquisition.data import checkerboard


def get_zod_mask_checker(C_checker: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    """
    Probe the system with a maximally-bright uniform grating (checkerboard
    period (1,1)) to separate the static ZOD from the deflected field controlled by the hologram.

    Parameters
    ----------
    C_checker : torch.Tensor
        The camera-plane intensity pattern resulting from the checkerboard probe.
        Use get_checker_holo() to generate the probe pattern and pass it through the system.
    threshold : float, optional
        The intensity threshold to identify the ZOD region. Default is 0.01.
    """
    device = C_checker.device
    mask = (C_checker[0] > threshold).cpu().numpy().astype(np.uint8)
    star = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    mask = cv2.dilate(mask, star, iterations=1)
    zod_mask = torch.from_numpy(mask).to(device=device, dtype=torch.bool)
    return (~zod_mask).float()

def get_zod_mask_median(C_all: torch.Tensor, threshold: float = 0.15) -> torch.Tensor:
    """
    Compute a ZOD mask by taking the median of multiple camera-plane intensity patterns.
    Use this if the other method (get_zod_mask_checker) is not suitable.

    Parameters
    ----------
    C_all : torch.Tensor
        A batch of camera-plane intensity patterns (B, H, W).
    threshold : float, optional
        The intensity threshold to identify the ZOD region. Default is 0.15.
    
    Example usage:
    -------------
    In simulation, we can generate a batch of random holograms and their corresponding 
    camera-plane images using the frozen ground truth model, and then compute the ZOD mask from the median pattern:

    batch_fn_random = make_sim_batch_fn(model_true, pattern="random")\n
    _, C_all = batch_fn_random(torch.arange(NUM_IMAGES))\n
    zod_mask = get_zod_mask_median(C_all, threshold=0.15)
    """
    median_pattern = torch.median(C_all, dim=0)
    mask = (median_pattern[0] > threshold).cpu().numpy().astype(np.uint8)
    star = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    mask = cv2.erode(mask, star, iterations=1)
    mask = cv2.dilate(mask, star, iterations=1)
    zod_mask = torch.from_numpy(mask).to(device=C_all.device, dtype=torch.bool)
    return (~zod_mask).float()
    