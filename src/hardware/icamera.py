"""
Abstract base class for scientific camera backends.
"""
from abc import ABC, abstractmethod


class ICamera(ABC):
    """Abstract base class for scientific cameras."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def open(self):
        """Establish the connection to the camera and apply the settings given at construction time."""

    @abstractmethod
    def close(self):
        """Close the connection to the camera."""

    def __enter__(self):
        """Open the device when entering a `with` block."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the device when exiting a `with` block."""
        self.close()

    # ------------------------------------------------------------------
    # Single-shot capture
    # ------------------------------------------------------------------

    @abstractmethod
    def capture(self):
        """Capture and return a single frame."""

    @abstractmethod
    def capture_hdr(self, num_frames):
        """Capture `num_frames` frames at successively longer exposures and merge them into one HDR frame."""

    # ------------------------------------------------------------------
    # Continuous (free-running) acquisition -- used to time-average many
    # frames without paying a software round-trip per frame (see
    # src/acquisition/capture.py's capture_multiframe).
    # ------------------------------------------------------------------

    @abstractmethod
    def start_acquisition(self):
        """Put the camera into continuous acquisition mode, for use with `get_frame`."""

    @abstractmethod
    def stop_acquisition(self):
        """Stop acquisition started by `start_acquisition`."""

    @abstractmethod
    def get_frame(self):
        """Return the next frame from an acquisition started by `start_acquisition`."""

    # ------------------------------------------------------------------
    # Settings (mutable, per-instance)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def exposure_time(self):
        """Exposure time in seconds (the last value set, not necessarily re-queried from hardware)."""

    @exposure_time.setter
    @abstractmethod
    def exposure_time(self, value):
        """Set the exposure time in seconds and apply it to the hardware."""

    @property
    @abstractmethod
    def gain(self):
        """Sensor gain in dB (the last value set, not necessarily re-queried from hardware)."""

    @gain.setter
    @abstractmethod
    def gain(self, value):
        """Set the sensor gain in dB and apply it to the hardware."""

    @abstractmethod
    def set_roi(self, x, y, width, height):
        """Set the region of interest, in pixels, on the full sensor."""

    # ------------------------------------------------------------------
    # Fixed hardware properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def shape(self):
        """Current (height, width) of a captured frame -- reflects the active ROI."""

    @staticmethod
    @abstractmethod
    def get_pixel_pitch():
        """Sensor pixel pitch, in microns."""

    @staticmethod
    @abstractmethod
    def get_range():
        """Maximum representable pixel value (i.e. grayscale range - 1)."""