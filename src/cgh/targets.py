"""
Target intensity pattern generation for CGH.
"""
import torch
from src.acquisition.data import checkerboard
from src.utils.utils import squish_image
from src.io.img_handler import load_svg


def get_target_circle(shape: tuple, r_in_frac: float = 0.4, r_out_frac: float = 0.8, device: str = "cpu") -> torch.Tensor:
    """
    Generate a circular-ring target intensity pattern of the given far-field shape.

    Parameters
    ----------
    shape : tuple
        The desired shape of the target intensity pattern (height, width).
    r_in_frac, r_out_frac : float, optional
        Inner/outer ring radius as a fraction of the half-frame radius.
        Defaults (0.4, 0.8) match the previous hardcoded ring.
    device : str, optional
        The device on which to create the tensor. Default is "cpu".
    """
    M, N = shape
    M_max = max(M, N)

    Y, X = torch.meshgrid(torch.arange(M_max), torch.arange(M_max), indexing='ij')
    center = (M_max // 2, M_max // 2)
    r = M_max // 2
    r_in, r_out = int(r_in_frac * r), int(r_out_frac * r)  # inner and outer radii for the ring
    d2 = (X - center[1]) ** 2 + (Y - center[0]) ** 2
    target = (d2 <= r_out ** 2).float()
    target[d2 <= r_in ** 2] = 0.0

    if (M_max, M_max) != shape:
        return torch.tensor(squish_image(target.detach().cpu().numpy(), (M, N)), device=device)
    else:
        return target


def get_target_checkerboard(shape: tuple, pitch=10, device: str = "cpu") -> torch.Tensor:
    """
    Generate a checkerboard target intensity pattern of the given far-field shape.

    Parameters
    ----------
    shape : tuple
        The desired shape of the target intensity pattern (height, width).
    pitch : int or tuple, optional
        Checkerboard square size in pixels. An int applies the same pitch to
        both axes (e.g. pitch=10 -> pitch=(10, 10)); pass a (row, col) tuple
        for non-square squares. Default is 10, matching the previous
        hardcoded pitch=(10, 10).
    device : str, optional
        The device on which to create the tensor. Default is "cpu".
    """
    M, N = shape
    M_max = max(M, N)
    pitch_tuple = (pitch, pitch) if isinstance(pitch, int) else tuple(pitch)

    target = checkerboard((M_max, M_max), pitch=pitch_tuple, device=device)

    if (M_max, M_max) != shape:
        return torch.tensor(squish_image(target.detach().cpu().numpy(), (M, N)), device=device)
    else:
        return target


def load_target_from_image(path: str, shape: tuple, device: str = "cpu") -> torch.Tensor:
    """
    Load a real target image (SVG/PNG) instead of a synthetic pattern.

    Parameters
    ----------
    path : str
        Path to the target image (SVG or PNG). Not resolved against
        DATA_ROOT here -- pass an already-resolved path, e.g.
        `data_path(cfg.target_path)`.
    shape : tuple
        The desired far-field shape (height, width) to squish the loaded image to.
    device : str, optional
        The device on which to create the tensor. Default is "cpu".
    """
    raw = load_svg(path, size=(max(shape), max(shape)))
    return torch.tensor(squish_image(raw, target_shape=shape), device=device)