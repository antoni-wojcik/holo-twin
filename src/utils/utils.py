"""
Different utility functions for image processing and manipulation, 
including phase to grayscale conversion, image resizing, squaring, rescaling, padding, 
cropping, and low-pass filtering.
These functions are used in the context of holographic imaging and training of HoloSystems.
"""

import numpy as np
from scipy.ndimage import zoom 

def squish_image(image: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Resize an image to a target shape using zoom.
    Args:
        image (np.ndarray): The input image to resize.
        target_shape (tuple): The desired shape of the output image (height, width).
    Returns:
        np.ndarray: The resized image.
    """
    # zoom might return negative values, so we take the absolute value
    return np.abs(zoom(image, (target_shape[0] / image.shape[0], target_shape[1] / image.shape[1])))

def square_image(image: np.ndarray, size=None) -> np.ndarray:
    """
    Make the image square by padding it to the specified size.
    If the image is larger than the specified size, it will be cropped.
    If the image is smaller, it will be padded with zeros.
    Parameters:
    - image: Input image as a 2D NumPy array.
    - size: Desired size for the squared image. If None, it will be set to the maximum dimension of the image.
    Returns:
    - image_new: Squared image as a 2D NumPy array.
    """

    if size is None:
        size = max(image.shape)

    pad_x = (size - image.shape[0]) // 2
    pad_y = (size - image.shape[1]) // 2
    image_new = np.pad(image, ((pad_x, size - image.shape[0] - pad_x),
                               (pad_y, size - image.shape[1] - pad_y)),
                       mode='constant', constant_values=0)
    
    return image_new

def rescale_to_camera(intensity: np.ndarray, cam_shape: tuple) -> np.ndarray:
    """
    Rescale the intensity to match the camera shape.
    The input intensity is assumed to be a square image.
    """
    new_h = int(intensity.shape[1] * cam_shape[0] / cam_shape[1])
    intensity_cam = intensity[(intensity.shape[0] - new_h) // 2:(intensity.shape[0] + new_h) // 2, :]
    
    # Scale the predicted intensity to match the camera shape
    scale_factor = (cam_shape[0] / intensity_cam.shape[0], cam_shape[1] / intensity_cam.shape[1])
    intensity_cam_resized = zoom(intensity_cam, scale_factor, order=3)
    
    return intensity_cam_resized

def pad(field: np.ndarray, scale=2) -> np.ndarray:
    """Pad the field to `scale` times the maximum dimension, centering the original field."""
    M, N = field.shape
    padded_size = max(M, N) * scale
    pad_y = (padded_size - M) // 2
    pad_x = (padded_size - N) // 2
    field_padded = np.pad(field, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant')
    return field_padded

def crop(field: np.ndarray, M: int, N: int, scale=2) -> np.ndarray:
    """Crop the centered field to the original size after padding."""
    padded_size = max(M, N) * scale
    pad_y = (padded_size - M) // 2
    pad_x = (padded_size - N) // 2
    field_cropped = field[pad_y:pad_y + M, pad_x:pad_x + N]
    return field_cropped

def lowpass_image(image: np.ndarray, slm_shape: tuple) -> np.ndarray:
    """
    Apply a low-pass filter to the image by zeroing out high-frequency components in the Fourier domain.
    The image is first transformed to the Fourier domain, then cropped and padded to the SLM shape, 
    and finally transformed back to the spatial domain. Assumes the input image is 2x the SLM shape.
    """
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))
    spectrum = pad(crop(spectrum, *slm_shape))
    image_pass = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(spectrum)))
    return np.abs(image_pass)