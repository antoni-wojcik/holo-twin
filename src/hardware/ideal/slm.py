from src.hardware.islm import ISLM
import numpy as np

class SLMIdeal(ISLM):
    def __init__(self, resolution, pixel_pitch, wavelength, grayscale_range=1024, in_wavefront=None):
        """
        Initialize the SLM with its type, resolution, and pixel pitch.

        :param resolution: Resolution of the SLM (e.g., (1920, 1080)).
        :param pixel_pitch: Pixel pitch of the SLM in microns.
        :param wavelength: Wavelength of the light in nm.
        :param grayscale_range: Grayscale range of the SLM (default is 1024).
        """
        self.resolution = resolution
        self.pixel_pitch = pixel_pitch
        self.wavelength = wavelength
        self.grayscale_range = grayscale_range

        # Initialize the display with zeros
        self.display(np.zeros(resolution, dtype=np.float32))

        # Add incident wavefront if specified
        if in_wavefront is not None:
            self.in_wavefront = in_wavefront
        else:
            self.in_wavefront = np.ones(resolution, dtype=np.complex128)

    def open(self):
        """ Open the SLM device. """
        # The ideal SLM does not require any specific opening procedure.
        pass

    def close(self):
        """ Close the SLM device. """
        # The ideal SLM does not require any specific closing procedure.
        pass

    def display(self, values):
        """
        Display a hologram on the SLM. This is to be used with the ideal Camera which processes the hologram.
        :param values: 2D NumPy array representing the image to be displayed.
        """
        self.phase = values.astype(np.float32) / self.grayscale_range * 2 * np.pi

    def get_shape(self):
        """
        Get the shape of the SLM.
        :return: Tuple representing the resolution of the SLM.
        """
        return self.resolution
    
    def get_pixel_pitch(self):
        """
        Get the pixel pitch of the SLM.
        :return: Pixel pitch of the SLM in microns.
        """
        return self.pixel_pitch