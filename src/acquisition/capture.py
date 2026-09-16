"""
Hardware display+capture loops. `acquire_batch` takes a `pattern_fn` (the
same kind of on-demand generator as make_sim_batch_fn's gen_fn) instead of
a precomputed (N, H, W) array -- one pattern is generated, converted,
displayed, and saved per iteration.
"""
from typing import Callable
import torch
import numpy as np
from tqdm import tqdm
import time
from src.io.data_io import ExpIO
from src.acquisition.data import to_grayscale_uint16


def acquire_batch(
    slm, camera,
    num_samples: int,
    pattern_fn: Callable[[int], torch.Tensor],
    exp: ExpIO, subdir_name: str,
    grayscale_range: int,
    start_idx: int = 0,
    save_images: bool = True
):
    """
    For i in range(num_samples): generate pattern_fn(i) (torch, [0,1]),
    convert to uint16, display, capture, save the pair via ExpIO.
    """
    for i in tqdm(range(num_samples), desc=f"Acquiring samples ({subdir_name})"):
        idx = start_idx + i
        holo_np = to_grayscale_uint16(pattern_fn(idx), grayscale_range)
        slm.display(holo_np)
        capture = camera.capture()
        exp.save_capture_pair(holo_np, capture, idx, subdir_name=subdir_name, save_images=save_images)

    print(f"Finished acquiring {num_samples} samples into '{subdir_name}'.")


def display_and_average(
        slm, camera,
        g_batch: torch.Tensor,
        grayscale_range: int,
        exp: ExpIO = None,
        subdir_name: str = None,
        save_images: bool = True
) -> np.ndarray:
    """
    Display each hologram in g_batch (N, H, W, torch [0,1]) sequentially and
    return the average capture. If `exp` is given, saves each individual
    capture plus the average under `subdir_name`.

    This is the TMBGD-style evaluation from citl_display_multibatch.py:
    g_batch is typically the output of CGH optimization (a small batch of
    holograms meant to be time-multiplexed), not a training dataset.
    """
    N = g_batch.shape[0]
    avg_capture = None
    for i in tqdm(range(N), desc="Displaying holograms"):
        holo_np = to_grayscale_uint16(g_batch[i], grayscale_range)
        slm.display(holo_np)
        capture = camera.capture()
        if exp is not None:
            exp.save_npy(capture, f"{i}", subdir_name=f"{subdir_name}/captures" if subdir_name else "captures")
            if save_images:
                exp.save_image(capture, f"{i}", subdir_name=f"{subdir_name}/captures/imgs" if subdir_name else "captures/imgs")
        avg_capture = capture.astype(np.float64) / N if avg_capture is None else avg_capture + capture.astype(np.float64) / N

    if exp is not None:
        exp.save_npy(avg_capture, "average", subdir_name=subdir_name)
        if save_images:
            exp.save_image(avg_capture, "average", subdir_name=f"{subdir_name}/captures/imgs" if subdir_name else "captures/imgs")
    return avg_capture


def capture_multiframe(
        slm, camera,
        g_batch: torch.Tensor,
        grayscale_range: int,
        exp: ExpIO = None,
        subdir_name: str = None,
        save_images: bool = True
) -> np.ndarray:

    """
    Display multiple holograms and capture a single time-averaged image.
    This is more realistic than post-processing the individual captures, since the camera integrates over time.
    """
    N = g_batch.shape[0]
    frames = [
        to_grayscale_uint16(g_batch[i], grayscale_range)
        for i in range(N)
    ]

    slm.load_memory(frames)
    period = camera.exposure_time / N
    slm.display(0)              # optional: start from known frame
    camera.start_acquisition()  # start the camera acquisition before the display loop

    start = time.perf_counter()
    for i in tqdm(range(N), desc="Displaying holograms"):
        slm.display_memory(i)

        target = start + (i + 1) * period
        while time.perf_counter() < target:
            pass

    capture = camera.get_frame()
    camera.stop_acquisition()  # stop the camera acquisition after the display loop

    if exp is not None:
        exp.save_npy(capture, "average", subdir_name=subdir_name)
        if save_images:
            exp.save_image(capture, "average", subdir_name=f"{subdir_name}/captures/imgs" if subdir_name else "captures/imgs")
    return capture