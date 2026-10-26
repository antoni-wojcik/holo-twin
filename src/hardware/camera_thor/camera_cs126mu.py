"""
Thorlabs CS126MU (Zelux) camera backend, via pylablib. This is the camera
currently used by the experimental pipelines.

Note: the original pylablib TLCamera code breaks on monochrome cameras;
see https://github.com/AlexShkarin/pyLabLib/issues/65 for the required
TLCamera.py patch.
"""
import numpy as np
import pylablib.devices.Thorlabs as tl

from src.hardware.icamera import ICamera

# --- hardware constants ---
CAMERA_SHAPE = (3000, 4096)      # Camera shape in pixels, full sensor
PIXEL_PITCH = 3.45               # Pixel pitch in micrometers
GRAYSCALE_RANGE = 4096           # 12-bit grayscale range (0-4095)
DEFAULT_ROI_SHAPE = (3000, 3000)  # Default ROI shape in pixels (height, width)


class CameraThorlabs(ICamera):
    """Camera control for the Thorlabs Zelux CS126MU/CS165MU family, via pylablib."""

    CAPTURE_TIMEOUT = 15  # Maximum time to wait for a single capture, in seconds

    def __init__(self, camera_num=0, exposure_time=0.01, gain=0, flip_ud=True, flip_lr=False, roi_shape=DEFAULT_ROI_SHAPE):
        """
        Parameters
        ----------
        camera_num : int
            Index into the list of detected Thorlabs cameras to open.
        exposure_time : float
            Exposure time in seconds, applied once the camera is opened.
        gain : float
            Sensor gain in dB, applied once the camera is opened.
        flip_ud, flip_lr : bool
            Flip captured frames vertically / horizontally (to correct for the camera's mounting orientation).
        roi_shape : tuple
            (height, width) of the region of interest to center on the sensor when opened.
        """
        self.camera_num = camera_num
        self._exposure_time = exposure_time
        self._gain = gain
        self.flip_ud = flip_ud
        self.flip_lr = flip_lr
        self.roi_shape = roi_shape

        self.width, self.height = 0, 0
        self.camera = None   # underlying pylablib camera handle, set by open()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self):
        """Detect and connect to the camera, push exposure/gain, and center the default ROI."""
        detected_camera = tl.list_cameras_tlcam()[self.camera_num]
        self.camera = tl.ThorlabsTLCamera(serial=detected_camera)
        self.camera.open()
        print('Camera opened:', self.camera.is_opened())

        self.camera.enable_led(False)

        # Push the settings requested at construction time down to the hardware now that it's open.
        self.gain = self._gain
        self.exposure_time = self._exposure_time

        self.width, self.height = self.camera.get_detector_size()
        self.camera.setup_acquisition(nframes=2)

        offset_x = (self.width - self.roi_shape[1]) // 2
        offset_y = (self.height - self.roi_shape[0]) // 2
        self.set_roi(offset_x, offset_y, width=self.roi_shape[1], height=self.roi_shape[0])

        # The actual ROI can differ slightly from requested (hardware increment constraints) --
        # read the true size back from a real frame.
        frame = self.camera.snap(timeout=self.CAPTURE_TIMEOUT)
        self.height, self.width = frame.shape

    def close(self):
        """Reset the ROI to the full sensor and close the connection."""
        self.set_roi(0, 0, width=self.width, height=self.height)
        self.camera.close()
        print('Camera closed:', not self.camera.is_opened())

    # ------------------------------------------------------------------
    # Single-shot capture
    # ------------------------------------------------------------------

    def capture(self):
        """Capture and return a single frame, flipped per `flip_ud`/`flip_lr`."""
        frame = self.camera.snap(timeout=self.CAPTURE_TIMEOUT).astype(np.double)
        return self._flip(frame)

    def capture_hdr(self, num_frames):
        """
        Capture `num_frames` frames at exposures increasing by 10x each
        step (starting from the current `exposure_time`), and merge them
        into one HDR frame via inverse-intensity weighting -- higher-
        exposure frames contribute more in dark areas. Restores the
        original exposure before returning. Result is normalized to [0, 1].
        """
        images = np.zeros((num_frames, self.height, self.width), dtype=np.float32)
        exposure_times = np.zeros(num_frames, dtype=np.float32)

        exposure_time_target = self._exposure_time
        for i in range(num_frames):
            exposure_times[i] = self.camera.get_exposure()
            images[i, :, :] = self.capture().astype(np.float32)
            exposure_time_target *= 10
            self.camera.set_exposure(exposure_time_target)

        self.exposure_time = self.exposure_time  # restore the original exposure via the property setter

        weights = 1.0 / np.clip(images, 1, GRAYSCALE_RANGE - 1)  # avoid division by zero; lower values weighted higher
        numerator = np.sum(weights * images / exposure_times[:, None, None], axis=0)
        denominator = np.sum(weights / exposure_times[:, None, None], axis=0)
        hdr = numerator / denominator

        return hdr / np.max(hdr)

    # ------------------------------------------------------------------
    # Continuous (free-running) acquisition
    # ------------------------------------------------------------------

    def start_acquisition(self):
        """Put the camera into continuous acquisition mode, for use with `get_frame`."""
        self.camera.start_acquisition()

    def stop_acquisition(self):
        """Stop acquisition started by `start_acquisition`."""
        self.camera.stop_acquisition()

    def get_frame(self):
        """Block until the next frame is available and return it, flipped per `flip_ud`/`flip_lr`."""
        self.camera.wait_for_frame()
        frame = self.camera.read_oldest_image().astype(np.double)
        return self._flip(frame)

    # ------------------------------------------------------------------
    # Settings (mutable, per-instance)
    # ------------------------------------------------------------------

    @property
    def exposure_time(self):
        """Exposure time in seconds (the last value set, not re-queried from hardware)."""
        return self._exposure_time

    @exposure_time.setter
    def exposure_time(self, value):
        """Set the exposure time in seconds (Zelux CS165MU1/M -- may be ms or us on other camera models)."""
        self._exposure_time = value
        if self.camera is not None:
            self.camera.set_exposure(value)

    @property
    def gain(self):
        """Sensor gain in dB (the last value set, not re-queried from hardware)."""
        return self._gain

    @gain.setter
    def gain(self, value):
        """Set the sensor gain in dB."""
        self._gain = value
        if self.camera is not None:
            self.camera.set_gain(value)

    def set_roi(self, x, y, width, height, verbose=False):
        """Set the region of interest, in pixels, on the full sensor (adjusted for `flip_ud`/`flip_lr`)."""
        if self.flip_lr:
            x = self.width - x - width
        if self.flip_ud:
            y = self.height - y - height

        if verbose:
            print(f"Setting Camera ROI to x={x}, y={y}, width={width}, height={height}")

        self.camera.set_roi(hstart=x, hend=x + width, vstart=y, vend=y + height)

    # ------------------------------------------------------------------
    # Fixed hardware properties
    # ------------------------------------------------------------------

    @property
    def shape(self):
        """Current (height, width) of a captured frame -- reflects the active ROI, (0, 0) before `open()`."""
        return (self.height, self.width)

    @staticmethod
    def get_pixel_pitch():
        return PIXEL_PITCH

    @staticmethod
    def get_range():
        return GRAYSCALE_RANGE - 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flip(self, frame):
        """Apply `flip_ud`/`flip_lr` to a captured frame. Shared by `capture()` and `get_frame()`."""
        if self.flip_ud:
            frame = np.flipud(frame)
        if self.flip_lr:
            frame = np.fliplr(frame)
        return frame