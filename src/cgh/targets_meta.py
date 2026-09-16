"""
Generate a pattern of pillars in a grid layout.
Used to simulate Huygens metasurface-like targets for CGH.
"""
import numpy as np
import torch

from src.utils.utils import squish_image


def get_pattern(
        shape: tuple,
        num_cells: int, 
        cell_size: int, 
        pillar_size: tuple, 
        meta_shape: str = 'circle', 
        device: torch.device = None
) -> np.ndarray:
    """
    Generate a pattern of pillars in a grid layout.

    Parameters
    ----------
    shape : tuple
        The shape of the output pattern (height, width).
    num_cells : int
        Number of cells in the grid (grid will be num_cells x num_cells).
    cell_size : int
        Size of each cell in pixels (cell will be cell_size x cell_size).
    pillar_size : tuple
        A tuple (min_size, max_size) specifying the range of pillar sizes in pixels.
    meta_shape : str, optional
        Shape of the pillars. Options are 'square', 'circle', 'plus', 'rot_rect', or 'random'. Default is 'circle'.
    device : torch.device, optional
        The device on which to create the output tensor. If None, defaults to CPU.

    Returns
    -------
    pattern : np.ndarray
        A 2D numpy array of shape (num_cells * cell_size, num_cells * cell_size) containing the generated pattern.
    """
    pillar_size_min, pillar_size_max = pillar_size
    # place pillars in a grid layout
    pattern = np.zeros((num_cells * cell_size, num_cells * cell_size), dtype=float)
    for i in range(num_cells):
        for j in range(num_cells):
            size = np.random.randint(pillar_size_min, pillar_size_max + 1)
            x = i * cell_size
            y = j * cell_size
            _place_pillar(pattern, x, y, size, cell_size, pillar_size_min, pillar_size_max, meta_shape)

    pattern = squish_image(pattern, shape)
    pattern = torch.tensor(pattern, device=device, dtype=torch.float)
    return pattern


def _get_pillar_square(size: int, cell_size: int) -> np.ndarray:
    """A single filled square pillar, centred in a cell_size x cell_size cell."""
    pillar = np.zeros((cell_size, cell_size), dtype=float)
    center = cell_size // 2
    radius = size // 2
    pillar[center - radius:center + radius, center - radius:center + radius] = 1
    return pillar


def _get_pillar_circle(size: int, cell_size: int) -> np.ndarray:
    """A single filled circular pillar, centred in a cell_size x cell_size cell."""
    pillar = np.zeros((cell_size, cell_size), dtype=float)
    center = cell_size // 2
    radius = size // 2
    y, x = np.ogrid[:cell_size, :cell_size]
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    pillar[mask] = 1
    return pillar


def _get_pillar_plus(size: int, cell_size: int) -> np.ndarray:
    """A single filled plus-shaped pillar, centred in a cell_size x cell_size cell."""
    pillar = np.zeros((cell_size, cell_size), dtype=float)
    center = cell_size // 2
    radius = size // 4
    pillar[center - 2 * radius:center + 2 * radius, center - radius:center + radius] = 1  # Horizontal bar
    pillar[center - radius:center + radius, center - 2 * radius:center + 2 * radius] = 1  # Vertical bar
    return pillar


def _get_pillar_rot_rect(size: int, cell_size: int, pillar_size_min: int, pillar_size_max: int) -> np.ndarray:
    """A single filled rectangular pillar at a random rotation, centred in a cell_size x cell_size cell."""
    angle = np.random.uniform(0, 2 * np.pi)
    pillar = np.zeros((cell_size, cell_size), dtype=float)
    center = cell_size // 2

    length = pillar_size_max // 2
    width = pillar_size_min // 2
    y, x = np.ogrid[:cell_size, :cell_size]
    x_rot = (x - center) * np.cos(angle) - (y - center) * np.sin(angle)
    y_rot = (x - center) * np.sin(angle) + (y - center) * np.cos(angle)

    # Create a rotated rectangle pillar
    mask = np.logical_and(np.abs(x_rot) <= length, np.abs(y_rot) <= width)
    pillar[mask] = 1
    return pillar


def _place_pillar(
    pattern: np.ndarray, x: int, y: int, size: int, cell_size: int,
    pillar_size_min: int, pillar_size_max: int, shape: str = 'square',
) -> None:
    """Draw one pillar of the given shape into `pattern` at cell origin (x, y), in place."""
    if shape == 'square':
        pillar = _get_pillar_square(size, cell_size)
    elif shape == 'circle':
        pillar = _get_pillar_circle(size, cell_size)
    elif shape == 'plus':
        pillar = _get_pillar_plus(size, cell_size)
    elif shape == 'rot_rect':
        pillar = _get_pillar_rot_rect(size, cell_size, pillar_size_min, pillar_size_max)
    elif shape == 'random':
        r = np.random.rand()
        if r < 1 / 2:
            pillar = _get_pillar_square(size, cell_size)
        else:
            pillar = _get_pillar_circle(size, cell_size)
    else:
        raise ValueError(f"Unknown shape '{shape}'. Use 'square', 'circle', 'plus', 'rot_rect', or 'random'.")
    pattern[x:x + cell_size, y:y + cell_size] = pillar