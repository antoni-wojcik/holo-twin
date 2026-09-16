from src.hardware.icamera import ICamera
# The ideal camera works only with the ideal SLM
from src.hardware.ideal.slm import SLMIdeal
import numpy as np
from matplotlib import pyplot as plt

class CameraIdeal(ICamera):
    def __init__(self, pixel_pitch, slm: SLMIdeal, threshold = 1.0, upscale=True, verbose=False):
        """
        Initialize the Camera with its type, resolution, and pixel pitch.

        :param resolution: Resolution of the camera (e.g., (1920, 1080)).
        :param pixel_pitch: Pixel pitch of the camera in microns.
        :param slm: Ideal SLM object to be used with the camera.
        :param threshold: Threshold value for the camera (default is 1, because the imagees are normalised to [0, 1]).
        """
        self.pixel_pitch = pixel_pitch
        self.slm = slm
        self.threshold = threshold
        self.upscale = upscale
        if upscale:
            self.resolution = (slm.resolution[0] * 2, slm.resolution[1] * 2)  # Ideal camera has double the resolution of the SLM
        else:
            self.resolution = slm.resolution

        self.verbose = verbose

        self.roi_set = False
        self.roi = (0, 0, self.resolution[0], self.resolution[1])  # Default ROI is the full resolution

    def open(self):
        """ Open the camera device. """
        # The ideal camera does not require any specific opening procedure.
        pass

    def close(self):
        """ Close the camera device. """
        # The ideal camera does not require any specific closing procedure.
        pass

    def capture(self):
        """
        Capture an image projected by the SLM.
        :return: 2D NumPy array representing the captured image.
        """
        
        phase = self.slm.phase
        # Calculate the near field using the phase. Include the incident wavefront
        field_near = np.exp(1j * phase) * self.slm.in_wavefront
        # Pad it to double the size with zeroes, keeping the original in the center
        if self.upscale:
            M, N = field_near.shape
            field_near = np.pad(field_near, ((M//2, M//2), (N//2, N//2)), mode='constant', constant_values=0)
        # Perform the Fourier transform to get the far field
        field_far = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_near)))

        # Calculate the intensity and normalise it
        intensity = np.abs(field_far) ** 2
        intensity = intensity / np.max(intensity)

        # Crop the image to the ROI
        if self.roi_set:
            x, y, width, height = self.roi
            intensity = intensity[y:y+height, x:x+width]

        # Apply thresholding
        intensity = np.minimum(intensity, self.threshold)

        if self.verbose:
            plt.imshow(intensity, cmap='gray')
            plt.show()

        return intensity

    def capture_hdr(self, num_frames):
        return self.capture()

    def set_roi(self, x, y, width, height):
        self.roi_set = True
        self.roi = (x, y, width, height)

    def get_pixel_pitch(self):
        """
        Get the pixel pitch of the camera.
        :return: Pixel pitch of the camera in microns.
        """
        return self.pixel_pitch
    
    def get_range(self):
        """
        Get the grayscale range of the camera.
        :return: Grayscale range of the camera.
        """
        return 1.0  # Ideal camera has a normalized range from 0 to 1
    
    def get_shape(self):
        """
        Get the shape (resolution) of the camera.
        :return: Tuple representing the resolution of the camera (height, width).
        """
        return self.resolution