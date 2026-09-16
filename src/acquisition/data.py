"""
Pattern generation + batch_fn factories, unified across simulation and
experimental use.
"""
import os
import torch
import numpy as np
from src.twin.model import HoloSystem


# ---------------------------------------------------------------------
# Pattern generators (single pattern, given a seed/index)
# ---------------------------------------------------------------------

def checkerboard(shape: tuple, pitch: tuple, device="cpu") -> torch.Tensor:
    """Checkerboard pattern in [0, 1], given pitch (h, w) in pixels."""
    P, Q = shape
    Y, X = torch.meshgrid(torch.arange(P, device=device), torch.arange(Q, device=device), indexing='ij')
    return (((X // pitch[1]) % 2) ^ ((Y // pitch[0]) % 2)).float()


def pitch_for_index(index: int) -> tuple:
    """
    Deterministically maps an integer index to a checkerboard pitch,
    alternating orientation and doubling in size every two indices:
    0->(2,4), 1->(4,2), 2->(4,8), 3->(8,4), 4->(8,16), ...
    """
    j = index // 2 + 1
    return (2 ** j, 2 ** (j + 1)) if index % 2 == 0 else (2 ** (j + 1), 2 ** j)

def grating_pattern(shape: tuple, index: int, modulation: float = 0.4, device="cpu") -> torch.Tensor:
    """Single grating (checkerboard at the pitch for `index`), scaled by modulation."""
    return checkerboard(shape, pitch_for_index(index), device=device) * modulation


def mask_pattern(shape: tuple, modulation: float = 1.0, device="cpu") -> torch.Tensor:
    """
    (1,1)-pitch checkerboard used to isolate the ZOD (see src/training/masks.py):
    an ideal pixel deflects all power away from the zeroth order under this
    pattern, so leftover energy at the center is exactly the ZOD leakage.
    """
    return checkerboard(shape, (1, 1), device=device) * modulation


def random_pattern(shape: tuple, seed: int, device="cpu") -> torch.Tensor:
    """Seeded random pattern in [0, 1] -- same seed always reproduces the same pattern."""
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return torch.rand(shape, generator=gen, device=device)


# ---------------------------------------------------------------------
# Simulation: batch_fn draws (g, I_true) from a frozen ground-truth model
# ---------------------------------------------------------------------

def make_sim_batch_fn(model_true: HoloSystem, pattern: str = "random"):
    """
    pattern: "random", "checkerboard", or "mask" (the (1,1)-pitch ZOD probe,
    see mask_pattern - always the same single pattern regardless of index,
    so a DataSource built from this only needs total_samples=1).
    """
    device = model_true.geometry.device
    slm_shape = model_true.geometry.slm_pixels
    if pattern == "random":
        gen_fn = lambda i: random_pattern(slm_shape, i, device=device)
    elif pattern == "checkerboard":
        gen_fn = lambda i: grating_pattern(slm_shape, i, device=device)
    elif pattern == "mask":
        gen_fn = lambda i: mask_pattern(slm_shape, device=device)
    else:
        raise ValueError(f"Unknown pattern {pattern!r}; use 'random', 'checkerboard', or 'mask'.")

    def batch_fn(idx: torch.Tensor):
        g = torch.stack([gen_fn(int(i)) for i in idx])
        with torch.no_grad():
            I_true = model_true(g)
        return g, I_true

    return batch_fn


# ---------------------------------------------------------------------
# Experimental: batch_fn loads acquired (holo, capture) pairs from disk
# ---------------------------------------------------------------------

def _crop_center(img: np.ndarray, crop_size: tuple, offset: tuple = (0, 0)) -> np.ndarray:
    H, W = img.shape
    ch, cw = crop_size
    y0 = max((H - ch) // 2 + offset[0], 0)
    x0 = max((W - cw) // 2 + offset[1], 0)
    return img[y0:y0 + ch, x0:x0 + cw]


def make_experimental_batch_fn(data_path: str, camera_shape: tuple, device: str, grayscale_range: int):
    """
    `data_path` must contain holos/<idx>.npy and captures/<idx>.npy, i.e.
    pass e.g. os.path.join(acquisition_root, "checkers") - exactly what
    src/acquisition/capture.py's acquire_batch() writes. Reads fresh from
    disk on every call rather than preloading, so GPU memory only ever
    holds one micro-batch at a time.
    """
    holos_dir = os.path.join(data_path, "holos")
    captures_dir = os.path.join(data_path, "captures")

    def batch_fn(idx: torch.Tensor):
        g_list, I_list = [], []
        for i in idx:
            i = int(i.item())
            holo = from_unint16_grayscale(np.load(os.path.join(holos_dir, f"{i}.npy")), grayscale_range)
            img = np.load(os.path.join(captures_dir, f"{i}.npy"))
            img = _crop_center(img, camera_shape)
            img = img / np.amax(img)
            g_list.append(holo)
            I_list.append(img)
        g = torch.from_numpy(np.stack(g_list)).float().to(device)
        I_true = torch.from_numpy(np.stack(I_list)).float().to(device)
        return g, I_true

    return batch_fn


def from_unint16_grayscale(pattern: np.ndarray, grayscale_range: int) -> np.ndarray:
    """Convert a uint16 array from slm.display() back to [0, 1] float."""
    return pattern.astype(np.float32) / (grayscale_range - 1)


def to_grayscale_uint16(pattern: torch.Tensor, grayscale_range: int) -> np.ndarray:
    """Convert a torch [0,1] pattern to the uint16 array slm.display() expects.
    The one place a torch pattern touches numpy on the way to hardware."""
    return (pattern.clamp(0, 1) * (grayscale_range - 1)).round().cpu().numpy(force=True).astype(np.ushort)