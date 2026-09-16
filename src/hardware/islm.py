"""
Abstract base class for spatial light modulator (SLM) backends.
"""
from abc import ABC, abstractmethod

import numpy as np


class ISLM(ABC):
    """Abstract base class for spatial light modulators."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def open(self):
        """Establish the connection to the SLM device."""

    @abstractmethod
    def close(self):
        """Close the connection to the SLM."""

    def __enter__(self):
        """Open the device when entering a `with` block."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the device when exiting a `with` block."""
        self.close()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    @abstractmethod
    def display(self, values):
        """Display an image (or, for memory-mode backends, a memory slot index) on the SLM."""

    def display_memory(self, memory_offset):
        """
        Display a frame previously uploaded via `load_memory`, by its slot
        offset. Only supported by SLMs with onboard memory (e.g.
        SLMSantec) -- override alongside `load_memory` to support it.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support memory-mode display.")

    def load_memory(self, frames, memory_start=1):
        """
        Upload a sequence of frames into the SLM's onboard memory, for
        fast sequential display via `display_memory`. Only supported by
        SLMs with onboard memory (e.g. SLMSantec).
        """
        raise NotImplementedError(f"{type(self).__name__} does not support memory-mode display.")

    def wait_for_ready(self, timeout=5):
        """
        Block until the SLM reports it is ready to accept a new frame, or
        `timeout` seconds elapse. Returns True once ready. A no-op that
        always returns True by default; override for hardware with an
        explicit readiness handshake (e.g. SLMSantec).
        """
        return True

    def change_wavelength(self, wavelength, modulation_range=200, save=False):
        """
        Update the SLM's phase-to-voltage calibration for a new
        wavelength (nm). `modulation_range` sets the target modulation
        depth, `save` persists it as the device's power-on default. A
        no-op by default; override for hardware that needs an explicit
        calibration step (e.g. SLMSantec, SLMLuna).
        """
        pass

    # ------------------------------------------------------------------
    # Fixed hardware properties
    # ------------------------------------------------------------------

    @staticmethod
    @abstractmethod
    def get_shape():
        """Return the (height, width) resolution of the SLM, in pixels."""

    @staticmethod
    @abstractmethod
    def get_pixel_pitch():
        """Return the pixel pitch of the SLM, in microns."""

    @staticmethod
    @abstractmethod
    def get_grayscale_range():
        """Return the number of distinct grayscale levels the SLM accepts (e.g. 1024 for 10-bit)."""