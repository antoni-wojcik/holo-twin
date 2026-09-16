"""
Image I/O helpers: load/save grayscale images, and rasterize SVGs to
grayscale arrays (via cairosvg if available, otherwise via Inkscape).
"""
import numpy as np
import cv2
from src.config import INSKAPE_PATH, CAIROSVG_AVAILABLE


def load_image(path: str) -> np.ndarray:
    """
    Load a grayscale image from file, normalized to [0, 1].

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    np.ndarray
        Grayscale image, normalized to [0, 1].
    """
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float32)
    image = image / np.amax(image)  # Normalize to [0, 1]
    return image


def save_image(image: np.ndarray, path: str) -> None:
    """
    Save a grayscale image to file, normalized to [0, 255] and converted to uint8.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image array (any float/int range) to save.
    path : str
        Destination file path.
    """
    image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image_uint8 = image_norm.astype(np.uint8)
    cv2.imwrite(path, image_uint8)


if CAIROSVG_AVAILABLE:
    import cairosvg

    def load_svg(svg_path: str, size: tuple = (128, 128)) -> np.ndarray:
        """
        Convert an SVG file to a grayscale 2D array, normalized to [0, 1].

        Parameters
        ----------
        svg_path : str
            Path to the SVG file.
        size : tuple, optional
            (height, width) to render the output image at. Default is (128, 128).

        Returns
        -------
        np.ndarray
            Grayscale image, normalized to [0, 1].
        """
        # Convert SVG to PNG in memory
        png_data = cairosvg.svg2png(url=svg_path, output_width=size[1], output_height=size[0])

        # Convert PNG binary data to a NumPy array
        nparr = np.frombuffer(png_data, np.uint8)

        # Decode PNG image using OpenCV
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        # Normalize to [0, 1] range
        normalized_image = image.astype(np.float32) / 255.0

        return normalized_image
else:
    import subprocess
    import tempfile
    import os

    def load_svg(svg_path: str, size: tuple = (128, 128)) -> np.ndarray:
        """
        Convert an SVG file to a grayscale 2D array, normalized to [0, 1].
        Uses Inkscape (must be installed and in PATH, or set INSKAPE_PATH).

        Parameters
        ----------
        svg_path : str
            Path to the SVG file.
        size : tuple, optional
            (height, width) to render the output image at. Default is (128, 128).

        Returns
        -------
        np.ndarray
            Grayscale image, normalized to [0, 1].
        """
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_png:
            tmp_png_path = tmp_png.name

        try:
            # Run Inkscape to convert SVG to PNG
            subprocess.run([
                INSKAPE_PATH,
                svg_path,
                '--export-type=png',
                f'--export-filename={tmp_png_path}',
                f'--export-width={size[1]}',
                f'--export-height={size[0]}'
            ], check=True)

            # Load the PNG as grayscale
            image = cv2.imread(tmp_png_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError("Failed to load PNG image converted by Inkscape.")

            # Normalize
            normalized_image = image.astype(np.float32)
            normalized_image = normalized_image / np.amax(normalized_image)  # Normalize to [0, 1]
            return normalized_image

        finally:
            # Clean up temporary PNG file
            if os.path.exists(tmp_png_path):
                os.remove(tmp_png_path)