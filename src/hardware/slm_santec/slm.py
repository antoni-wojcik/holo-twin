"""
Santec SLM backend: controls a Santec LCOS-SLM over the vendor's ctypes
SDK (src/hardware/slm_santec/_slm_win.py), either via direct DVI frame
writes or via the device's onboard memory (up to MAX_MEMORY slots), for
fast sequential display of a pre-uploaded sequence.
"""
import ctypes
import sys
import time

import numpy as np

from src.hardware.islm import ISLM

if sys.platform.startswith("win"):
    from src.hardware.slm_santec import _slm_win as slmc
else:
    slmc = None
    print("Warning: Santec SLM not supported on non-Windows platforms.")

# --- hardware constants ---
SLM_SHAPE = (1200, 1920)   # SLM shape in pixels
PIXEL_PITCH = 8             # Pixel pitch in micrometers
GRAYSCALE_RANGE = 1024      # Santec SLM supports 10-bit grayscale, so the range is 0-1023


class SLMSantec(ISLM):
    """Spatial Light Modulator control via the Santec SLM SDK."""

    # --- Santec SDK color/status flags (see the vendor SDK docs) ---
    FLAGS_COLOR_R = 0x00000001
    FLAGS_COLOR_G = 0x00000002
    FLAGS_COLOR_B = 0x00000004
    FLAGS_COLOR_GRAY = 0x00000008
    FLAGS_RATE120 = 0x20000000

    SLM_OK = 0   # Status OK
    SLM_NG = 1   # Status Not Good
    SLM_BS = 2   # Status Busy
    SLM_ER = 3   # Status Error

    SLM_MODE_MEMORY = 0
    SLM_MODE_DVI = 1

    RATE_120HZ = True    # enable the 120Hz refresh-rate flag on open()
    MAX_MEMORY = 128     # number of onboard memory slots

    def __init__(self, wavelength=658, slm_number=1, use_memory_mode=False, set_wavelength=True, timeout=10):
        """
        Parameters
        ----------
        wavelength : float
            Calibration wavelength in nm, used to set up the phase-to-voltage LUT (see `change_wavelength`).
        slm_number : int
            The SLM's control-channel number. Only matters if multiple SLMs are connected.
        use_memory_mode : bool
            If True, `display()` shows a previously-uploaded memory slot (see `load_memory`); if False (default),
            `display()` writes a fresh frame directly over DVI.
        set_wavelength : bool
            If True, `open()` pushes `wavelength` to the SLM's calibration table.
        timeout : float
            Seconds `display()` waits for the SLM to report ready, before and after writing a frame.
        """
        self.use_memory_mode = use_memory_mode
        self._wavelength = wavelength
        self.SLMNumber = slm_number
        self.set_wavelength = set_wavelength
        self.timeout = timeout

        self.width = 0
        self.height = 0
        self.DisplayNumber = None   # set once open() finds the display; also doubles as the "is open" flag
        self.Flags = 0

        # Close any previous connection using this SLM number before we start.
        slmc.SLM_Ctrl_Close(self.SLMNumber)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, slm_display_number=2):
        """
        Detect the SLM's display, open the USB control + display
        connections, select memory/DVI mode, and (if `set_wavelength`)
        push the calibration wavelength.

        `slm_display_number` overrides the display auto-detection below --
        pass it if the display name lookup ever picks the wrong monitor.
        """
        print('SLM connecting')
        width = ctypes.c_ushort(0)
        height = ctypes.c_ushort(0)
        display_name = ctypes.create_string_buffer(64)

        slm_found = False
        for display_number in range(1, 8):
            ret = slmc.SLM_Disp_Info2(display_number, width, height, display_name)
            if ret == self.SLM_OK:
                names = display_name.value.decode('mbcs').split(',')
                if names[0] in 'LCOS-SLM' or display_number == slm_display_number:
                    # The display name lookup is occasionally wrong -- slm_display_number lets the
                    # caller override it and pick a display directly.
                    self.width = width.value
                    self.height = height.value
                    self.DisplayNumber = display_number
                    print(names, self.width, self.height)
                    slm_found = True
                    break

        if not slm_found:
            print('No SLM found')
            return

        self.Flags = self.FLAGS_RATE120 if self.RATE_120HZ else 0

        slmc.SLM_Disp_Open(self.DisplayNumber)
        slmc.SLM_Ctrl_Open(self.SLMNumber)

        mode = self.SLM_MODE_MEMORY if self.use_memory_mode else self.SLM_MODE_DVI
        slmc.SLM_Ctrl_WriteVI(self.SLMNumber, mode)

        if self.set_wavelength:
            self.change_wavelength(self._wavelength)

        print('SLM connected')

    def close(self):
        """Stop the SLM and close the USB control + display connections."""
        slmc.SLM_Ctrl_WriteDB(self.SLMNumber)
        slmc.SLM_Ctrl_Close(self.SLMNumber)
        slmc.SLM_Disp_Close(self.DisplayNumber)
        print('SLM closed')

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @property
    def wavelength(self):
        """Calibration wavelength in nm currently applied to the SLM."""
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        """Equivalent to `change_wavelength(value)` with default modulation_range/save."""
        self.change_wavelength(value)

    def change_wavelength(self, wavelength, modulation_range=200, save=False):
        """
        Write `wavelength` (nm) to the SLM's phase-to-voltage calibration
        table -- target modulation depth is `2*pi * modulation_range / 200`.
        If `save`, also persists it as the SLM's power-on default.
        """
        print("Changing SLM calibration wavelength to", wavelength, "nm")
        slmc.SLM_Ctrl_WriteWL(self.SLMNumber, wavelength, modulation_range)
        if save:
            slmc.SLM_Ctrl_WriteAW(self.SLMNumber)
        self._wavelength = wavelength

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def display(self, values_or_memory_offset):
        """
        Display an image on the SLM, waiting for the SLM to report ready
        before and after (up to `self.timeout` seconds each way).
        If `use_memory_mode`, `values_or_memory_offset` is a memory slot
        number (0-127, see `load_memory`); otherwise it's a 2D numpy array
        written directly over DVI.
        """
        self.wait_for_ready(timeout=self.timeout)
        if self.use_memory_mode:
            self.display_memory(values_or_memory_offset)
        else:
            self._display_dvi(values_or_memory_offset)
        self.wait_for_ready(timeout=self.timeout)

    def _display_dvi(self, values):
        """Write a 2D uint16 array directly to the SLM over DVI. Internal -- called by `display()`."""
        n_h, n_w = values.shape
        if n_w != self.width or n_h != self.height:
            print(f"Error: Wrong hologram dimensions: ({n_h}, {n_w})")
            return

        values_flat = values.flatten().astype(np.ushort)
        values_ctypes = (ctypes.c_ushort * len(values_flat))(*values_flat)
        slmc.SLM_Disp_Data(self.DisplayNumber, n_w, n_h, self.Flags, values_ctypes)

    def display_memory(self, memory_offset):
        """Display the frame previously uploaded to memory slot `1 + memory_offset` (see `load_memory`)."""
        memory_number = 1 + memory_offset
        ret = slmc.SLM_Ctrl_WriteDS(self.SLMNumber, memory_number)
        if ret != self.SLM_OK:
            print(f"Error displaying memory slot {memory_number}")

    def load_memory(self, frames, memory_start=1):
        """
        Upload a sequence of 2D grayscale frames into onboard memory
        slots `memory_start, memory_start + 1, ...`, for fast sequential
        display via `display_memory`.
        """
        if not frames:
            print("No frames to display")
            return

        num_frames = len(frames)
        if num_frames > self.MAX_MEMORY - memory_start + 1:
            print("Warning: Too many frames. Only first 128 slots will be used.")
        print(f"Loading {min(num_frames, self.MAX_MEMORY)} frames into memory")

        for i, frame in enumerate(frames[:self.MAX_MEMORY]):
            print(f"Loading frame {i} into memory slot {memory_start + i}")
            memory_number = memory_start + i
            n_h, n_w = frame.shape

            if n_w != self.width or n_h != self.height:
                print(f"Skipping frame {i}: Incorrect dimensions {frame.shape}")
                continue

            frame_flat = frame.flatten().astype(np.ushort)
            frame_ctypes = (ctypes.c_ushort * len(frame_flat))(*frame_flat)

            ret = slmc.SLM_Ctrl_WriteMI(self.SLMNumber, memory_number, n_w, n_h, 0, frame_ctypes)
            if ret != self.SLM_OK:
                print(f"Error loading frame {i} into memory slot {memory_number}")
                return

        print("Loaded the frames into memory")

    def wait_for_ready(self, timeout=5):
        """Poll the SLM's status until it reports ready, busy for `timeout` seconds, or an error occurs."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = slmc.SLM_Ctrl_ReadSU(self.SLMNumber)
            if status == self.SLM_OK:
                return True
            elif status == self.SLM_BS:
                time.sleep(0.01)  # busy -- check again in 10ms
            else:
                print(f"SLM error: status {status}")
                return False

        print("WARNING: SLM did not become ready within the timeout period.")
        return True

    # ------------------------------------------------------------------
    # Fixed hardware properties
    # ------------------------------------------------------------------

    @staticmethod
    def get_shape():
        return SLM_SHAPE

    @staticmethod
    def get_pixel_pitch():
        return PIXEL_PITCH

    @staticmethod
    def get_grayscale_range():
        return GRAYSCALE_RANGE


if __name__ == '__main__':
    with SLMSantec() as S:
        pass
    print("It works!")