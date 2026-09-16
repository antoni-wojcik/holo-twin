"""
Holographic digital twin.

Architecture overview
---------------------
OpticsGeometry  : plain dataclass; physical constants + derived grid dims.
                  Passed to every module so each module only carries what it needs.
                  Specified by the user, never trained. Immutable after construction.

PhysicalModule  : nn.Module base class for every optics / camera sub-module.
                  Provides a unified enable/disable and freeze/unfreeze API so
                  the parent (HoloSystem) never has to inspect flags in its forward().
                  Also provides a rebuild() method to reinitialise internal buffers 
                  if the geometry or module parameters change.

HoloSystem      : nn.Module that chains together all the sub-modules in the correct order.
                  Provides a unified forward() that takes a grayscale input and returns the camera output, 
                  as well as control methods to enable/disable/freeze/unfreeze each sub-module. 
                  It also provides a save() and load() method to persist the entire system state, 
                  including all module parameters and buffers. Also comes with a report() method 
                  to visualise the current state of the system.           

Module hierarchy
----------------
HoloSystem
├── lut        : LUTModule          — grayscale → phase (monotone LUT + optional depth map)
├── pixel      : PixelModule        — sub-pixel LC crosstalk via Gaussian kernel convolution (+ residual) and far-field envelope
├── slm_field  : SLMFieldModule     — learnable effective incident field (amplitude + phase)
├── pupil      : PupilModule        - isoplanatic aberration via Seidel coefficients + optional vignetting from finite lens aperture
├── background : BackgroundModule   — stray-light complex field in the far-field
└── camera     : CameraModule       — affine warp + dynamic-range clipping
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from typing import Optional

from matplotlib import patches
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter

# =============================================================================
# Data containers
# =============================================================================

@dataclass(frozen=True)
class AberrCoefficients:
    """
    Third-order (Seidel) aberration coefficients in radians.

    Attributes
    ----------
    coma, astigmatism, field_curvature, distortion, tilt_x, tilt_y : float
    """
    coma:            float = 0.0
    astigmatism:     float = 0.0
    field_curvature: float = 0.0
    distortion:      float = 0.0
    tilt_x:          float = 0.0
    tilt_y:          float = 0.0

    NUM_COEFFS: int = 6

    NAMES: tuple = dataclasses.field(
        default=("Coma", "Astigmatism", "Field curv.", "Distortion", "X-tilt", "Y-tilt"),
        init=False, repr=False, compare=False,
    )

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        return torch.tensor(
            [self.coma, self.astigmatism, self.field_curvature,
             self.distortion, self.tilt_x, self.tilt_y],
            dtype=torch.float, device=device,
        )

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "AberrCoefficients":
        return cls(*t.tolist())

@dataclass(frozen=True)
class OpticsGeometry:
    """
    Immutable physical description of the SLM + far-field + camera system.

    All derived integer grid sizes are computed once in __post_init__.
    This object is NOT an nn.Module — it carries no parameters.

    Attributes (inputs)
    -------------------
    slm_pixels       : (H, W) in pixels
    scale            : far-field oversampling factor
    fov              : field-of-view expansion (requires pixel module active)
    square_far_field : if True, far-field grid is square

    slm_pixel_pitch  : µm
    electrode_width  : µm — square electrode side; sets the fill factor F = (width/pitch)^2
    wavelength       : nm
    focal_length     : mm
    lens_diameter    : inch

    camera_shape          : (H, W) in pixels
    camera_pixel_pitch    : µm
    camera_affine_supersample : anti-aliasing upsampling factor for the affine warp
    camera_exposure_time      : seconds. Exposure time used when acquiring the training data for the twin
    affine_rotation_range : max rotation in radians (tanh-constrained). 
    affine_scale_range    : max relative scale deviation (tanh-constrained)
    affine_shift_range    : max shift in pixels (tanh-constrained)

    device                : "cuda" or "cpu" (auto-select if None)

    Attributes (derived, not passed to __init__)
    --------------------------------------------
    P, Q     : SLM pixel counts
    M, N     : first-order far-field sample counts
    Pf, Qf   : SLM samples giving the extended FOV
    Mf, Nf   : far-field samples in the extended FOV

    Attributes (constants)
    ----------------------
    MM_PER_INCH : mm per inch
    """

    # ---- physical ----
    slm_pixels:       tuple = (128, 128)
    scale:            int   = 2
    fov:              float = 1.0
    square_far_field: bool  = False

    slm_pixel_pitch:   float = 8.0      # µm
    electrode_width:  float = 7.8       # µm, square electrode; fill factor = (width/pitch)^2
    wavelength:       float = 632.8     # nm
    focal_length:     float = 100.0     # mm
    lens_diameter:    float = 1.0       # inch

    camera_shape:              tuple = (128, 128)
    camera_pixel_pitch:         float = 3.45    # µm
    camera_saturation_electrons: float = 10000.0
    camera_quantum_efficiency:   float = 0.55
    camera_exposure_time:        float = 1.0   # s
    camera_affine_supersample:   int   = 2

    affine_rotation_range: float = math.radians(5.0)  # radians
    affine_scale_range:    float = 0.05
    affine_shift_range:    float = 35.0   # pixels

    device: str = None # if None, auto-select "cuda" if available else "cpu"

    # ---- constants ----
    MM_PER_INCH: float = 25.4   # mm per inch

    # ---- derived ----
    P:  int = field(init=False)
    Q:  int = field(init=False)
    M:  int = field(init=False)
    N:  int = field(init=False)
    Pf: int = field(init=False)
    Qf: int = field(init=False)
    Mf: int = field(init=False)
    Nf: int = field(init=False)

    def __post_init__(self):
        self._find_device()
        self._build_grid()
        self._validate_pixel()

    def _validate_pixel(self):
        w = self.electrode_width / self.slm_pixel_pitch
        if not 0.0 < w <= 1.0:
            raise ValueError(
                f"electrode_width ({self.electrode_width} um) must be positive and no larger "
                f"than slm_pixel_pitch ({self.slm_pixel_pitch} um)."
            )
        if w * w < 0.5:
            # the pixel-envelope normalisation F + (1-F)rho is bounded below by
            # 2F-1, so a fill factor under 1/2 lets it vanish for rho near -1
            warnings.warn(
                f"Fill factor {w*w:.3f} is below 0.5; the pixel-envelope normalisation "
                f"can become ill-conditioned for strongly negative deadspace reflectance."
            )

    def _find_device(self):
        if self.device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            super().__setattr__("device", device)

    def _build_grid(self):
        P, Q = self.slm_pixels
        if self.square_far_field:
            M = N = self.scale * max(P, Q)
        else:
            M, N = self.scale * P, self.scale * Q
        Pf = int(self.fov * P)
        Qf = int(self.fov * Q)
        Mf = int(self.fov * M)
        Nf = int(self.fov * N)

        super().__setattr__("P", P)
        super().__setattr__("Q", Q)
        super().__setattr__("M", M)
        super().__setattr__("N", N)
        super().__setattr__("Pf", Pf)
        super().__setattr__("Qf", Qf)
        super().__setattr__("Mf", Mf)
        super().__setattr__("Nf", Nf)

    # --- conveniences ---
    @property
    def electrode_width_norm(self) -> float:
        """Electrode width in pixel-pitch units, fx = fy."""
        return self.electrode_width / self.slm_pixel_pitch

    @property
    def fill_factor(self) -> float:
        """Pixel fill factor F = fx*fy, from the electrode width and pitch."""
        return self.electrode_width_norm ** 2

    @property
    def slm_samples(self) -> tuple:    return (self.Pf, self.Qf)
    @property
    def far_first_samples(self) -> tuple: return (self.M,  self.N)
    @property
    def far_fov_samples(self) -> tuple:   return (self.Mf, self.Nf)

    def saturation_irradiance(self) -> float:
        """
        Absolute saturation irradiance in mW/m² for the camera.
        Derived from electron well depth, QE, pixel pitch, wavelength, and exposure.
        """
        hc = 1.9864e-1  # h*c in units compatible with mW/m², nm, µm², s
        return (self.camera_saturation_electrons * hc
                / (self.wavelength * self.camera_quantum_efficiency
                   * self.camera_pixel_pitch ** 2 * self.camera_exposure_time))


# =============================================================================
# PhysicalModule — unified base class
# =============================================================================

class PhysicalModule(nn.Module):
    """
    Base class for every differentiable physical module in the holographic twin.

    Enable / disable
    ----------------
    module.disable()  → forward() delegates to _forward_disabled() instead of
                        _forward_enabled().  Each subclass defines _forward_disabled()
                        to return the physically correct "ideal / no correction"
                        output for that module — e.g. LUTModule returns a linear
                        phase ramp, PixelModule returns exp(iφ), etc.
    module.enable()   → normal forward behaviour via _forward_enabled().

    Freeze / unfreeze  (training efficiency)
    ----------------------------------------
    module.freeze()   → calls _build_cached_buffers() to precompute all parameter-dependent tensors,
                        sets requires_grad=False on all parameters, switches to
                        eval mode.  Subsequent forward() calls skip recomputation.
    module.unfreeze() → restores training mode and re-enables gradients, 
                        calls _clear_cached_buffers() to delete cached tensors.

    These compose correctly with PyTorch's own .train() / .eval():
      system.eval() propagates through all children via nn.Module's standard
      mechanism, invoking our overridden train(mode=False) which clears _cached.

    Rebuilding
    ---------
    module.rebuild()  → calls _build_static_buffers() to rebuild geometry-only buffers,
                        and if the module is frozen, also calls _clear_cached_buffers()
                        and _build_cached_buffers() to rebuild parameter-dependent buffers.

    Subclass contract  (all four methods should be implemented)
    -----------------
    _build_static_buffers()
        Called once at __init__.  Register buffers that depend only on geometry,
        never on learned parameters.

    _build_cached_buffers()
        Precompute all parameter-dependent tensors. Called once on freeze().

    _clear_cached_buffers()
        Delete all parameter-dependent tensors. Called once on unfreeze().

    _forward_enabled(*args, **kwargs)
        The actual learnable physics — runs when the module is active.

    _forward_disabled(*args, **kwargs)
        The ideal / trivial physics — runs when the module is disabled.
        Default implementation raises NotImplementedError; every subclass
        must provide a physically meaningful fallback.

    Do NOT override forward() in subclasses.
    """

    def __init__(self, geometry: OpticsGeometry):
        super().__init__()
        self.geometry = geometry
        self._active: bool = True
        self._cached: bool = False
        self._build_static_buffers()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _build_static_buffers(self):
        """Register geometry-only buffers. Called once at init."""
        pass

    def _build_cached_buffers(self):
        """Register parameter-dependent buffers. Called once at freeze()."""
        pass

    def _clear_cached_buffers(self):
        """Delete parameter-dependent buffers. Called once at unfreeze()."""
        pass

    def _forward_enabled(self, *args, **kwargs):
        """Learnable physics. Must be implemented by every subclass."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _forward_enabled()"
        )

    def _forward_disabled(self, *args, **kwargs):
        """
        Ideal / trivial physics used when the module is disabled.
        Every subclass must implement this with a physically meaningful
        fallback (e.g. linear LUT, identity field, zero background).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _forward_disabled(). "
            "This should return the physically correct output when this "
            "module's learned correction is not applied."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enable(self) -> "PhysicalModule":
        """Activate this module's learned contribution. Returns self for chaining."""
        self.unfreeze()
        self._active = True
        return self

    def disable(self) -> "PhysicalModule":
        """
        Deactivate this module's learned correction.  forward() will call
        _forward_disabled() which must return the physically correct
        ideal/trivial output. Returns self for chaining.
        """
        self.freeze()
        self._active = False
        return self

    @property
    def is_active(self) -> bool:
        return self._active
    
    @property
    def is_frozen(self) -> bool:
        return self._cached

    def freeze(self) -> "PhysicalModule":
        """
        Pre-cache all parameter-dependent tensors and stop gradient flow.
        Safe to call on individual modules while siblings remain trainable.
        Returns self for chaining.
        """
        self.eval()
        with torch.no_grad():
            self._build_cached_buffers()
        self._cached = True
        self.requires_grad_(False)
        return self

    def unfreeze(self) -> "PhysicalModule":
        """Return to training mode"""
        self.train()
        self._clear_cached_buffers()
        self._cached = False
        self.requires_grad_(True)
        return self

    def train(self, mode: bool = True) -> "PhysicalModule":
        """Override so that switching back to train mode clears the cache."""
        super().train(mode)
        if mode:
            self._cached = False
            self._clear_cached_buffers()
        return self

    def rebuild(self) -> "PhysicalModule":
        """
        Rebuild the internal buffers. 
        Useful if the parameter arguments have changed.
        """

        # write over the existing buffers with new ones
        self._build_static_buffers()

        # rebuild the cached buffers if the module is frozen
        if self.is_frozen:
            self._clear_cached_buffers()
            self._build_cached_buffers()
        return self

    # ------------------------------------------------------------------
    # forward — do NOT override in subclasses
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        if not self._active:
            return self._forward_disabled(*args, **kwargs)

        return self._forward_enabled(*args, **kwargs)


# =============================================================================
# Utilities: centred 2-D FFT with zero-padding, 
# complex interpolation, and softplus inverse
# =============================================================================

def _fft2_padded(field: torch.Tensor, out_shape: tuple) -> torch.Tensor:
    """
    Zero-pad *field* to *out_shape* and compute a centred 2-D FFT.

    Parameters
    ----------
    field     : (..., P, Q)
    out_shape : (M, N)

    Returns
    -------
    Tensor (..., M, N) complex
    """
    M, N = out_shape
    *batch, P, Q = field.shape
    pad_top    = (M - P) // 2
    pad_bottom = M - P - pad_top
    pad_left   = (N - Q) // 2
    pad_right  = N - Q - pad_left
    field_pad  = F.pad(field, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(field_pad), norm="ortho")
    )

def _centred_fft_axis(
    x: torch.Tensor,
    dim: int,
    L_out: int,
    n_crop: int,
    chunk: Optional[int] = None,
) -> torch.Tensor:
    """
    Centred, zero-padded 1-D FFT along *dim*, cropped to *n_crop* bins about DC.

    Mathematically identical to::

        xp  = centre-zero-pad(x, length L_out, along dim)
        Xf  = fftshift(fft(ifftshift(xp, dim), dim), dim)
        out = Xf.narrow(dim, (L_out - n_crop) // 2, n_crop)

    but never materialises the shifted arrays: the ifftshift is folded into
    where the samples are written in the padded workspace, and the fftshift is
    folded into a wrapped narrow of the *cropped* output only.  That removes
    two full-length roll/copy passes over an (L_out x ...) tensor.

    Unnormalised (no ``norm=`` division) -- the caller applies the scaling.

    Parameters
    ----------
    x : (..., S, ...)
        Input tensor, with the transform dimension at index *dim*.
    dim : int
        Dimension along which to compute the FFT.
    L_out : int
        Length of the zero-padded FFT workspace along *dim*.
    n_crop : int
        Number of output bins to return, centred about DC. Must be <= L_out.
    chunk : int, optional
        If provided, process *chunk* lines at a time along the other axis
        (the one not being transformed).  This is useful if the other axis is
        large and the FFT workspace would exceed available memory.  If None,
        process all lines at once.
    """
    dim = dim % x.ndim
    S = x.shape[dim]
    if L_out < S:
        raise ValueError(f"L_out ({L_out}) must be >= input length ({S})")

    lo    = (L_out - S) // 2                     # centred placement of x
    start = (lo - L_out // 2) % L_out            # ...after ifftshift
    c0    = (L_out - n_crop) // 2                # centred crop offset
    s     = (c0 - L_out // 2) % L_out            # ...after fftshift

    def _one(blk: torch.Tensor) -> torch.Tensor:
        if L_out == S and start == 0:
            buf = blk
        else:
            shape = list(blk.shape)
            shape[dim] = L_out
            buf = blk.new_zeros(shape)
            n1 = min(S, L_out - start)
            buf.narrow(dim, start, n1).copy_(blk.narrow(dim, 0, n1))
            if n1 < S:
                buf.narrow(dim, 0, S - n1).copy_(blk.narrow(dim, n1, S - n1))
        Xf = torch.fft.fft(buf, dim=dim)
        if n_crop > L_out:
            # Requested window is wider than one DFT period. Gather modulo L_out.
            idx = (torch.arange(n_crop, device=Xf.device) + s) % L_out
            return Xf.index_select(dim, idx)
        m1 = min(n_crop, L_out - s)
        head = Xf.narrow(dim, s, m1)
        if m1 == n_crop:
            return head.clone()                  # clone: release the L_out buffer
        return torch.cat([head, Xf.narrow(dim, 0, n_crop - m1)], dim=dim)

    other = x.ndim - 1 if dim == x.ndim - 2 else x.ndim - 2
    n_other = x.shape[other]
    if chunk is None or chunk >= n_other:
        return _one(x)
    return torch.cat(
        [_one(x.narrow(other, i, min(chunk, n_other - i)))
         for i in range(0, n_other, chunk)],
        dim=other,
    )


def _auto_chunk(n_lines: int, line_bytes: int, budget_bytes: int) -> int:
    """
    Number of lines to process per block so that one block's workspace stays
    near *budget_bytes*.  Returns at least 1 and at most n_lines.
    """
    if budget_bytes <= 0 or line_bytes <= 0:
        return n_lines
    n = max(1, budget_bytes // line_bytes)
    if n >= n_lines:
        return n_lines
    # even split, so the last block is not a tiny tail
    n_blocks = (n_lines + n - 1) // n
    return (n_lines + n_blocks - 1) // n_blocks


def _resize_circular(x: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """
    Resize the last two dims of x to target_shape, treating x as exactly
    periodic: wraps via circular padding when a dimension needs to grow,
    centre-crops when it needs to shrink.
    """
    *_, H, W = x.shape
    Ht, Wt = target_shape

    # Grow (pad) any dimension that needs it, in a single combined circular
    # pad call -- mirrors the exact call pattern the old (working) circular
    # pad used; a 0-length pad on an axis that doesn't need to grow is a no-op.
    pad_top = pad_bot = pad_left = pad_right = 0
    if Ht > H:
        pad_top = (Ht - H) // 2
        pad_bot = Ht - H - pad_top
    if Wt > W:
        pad_left = (Wt - W) // 2
        pad_right = Wt - W - pad_left
    if pad_top or pad_bot or pad_left or pad_right:
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bot), mode='circular')

    # Then centre-crop any dimension that needs to shrink.
    *_, H2, W2 = x.shape
    if Ht < H2 or Wt < W2:
        crop_top  = (H2 - Ht) // 2 if Ht < H2 else 0
        crop_left = (W2 - Wt) // 2 if Wt < W2 else 0
        x = x[..., crop_top:crop_top + Ht, crop_left:crop_left + Wt]
    return x

def _interpolate_complex_phasor(
    A: torch.Tensor,
    size: tuple,
    mode: str = "bicubic",
) -> torch.Tensor:
    """Interpolate a 2-D complex tensor by splitting into real/imag channels."""
    A4 = torch.view_as_real(A.resolve_conj()).permute(2, 0, 1).unsqueeze(0)
    A_up = F.interpolate(A4, size=size, mode=mode, align_corners=False)
    return torch.view_as_complex(A_up.squeeze(0).permute(1, 2, 0).contiguous())

def _interpolate_real(
    A: torch.Tensor,
    size: tuple,
    mode: str = "bicubic",
) -> torch.Tensor:
    """Interpolate a 2-D real tensor."""
    A_up = F.interpolate(A.unsqueeze(0).unsqueeze(0), size=size, mode=mode, align_corners=False)
    return A_up.squeeze(0).squeeze(0)

def _interpolate_complex(
    A: torch.Tensor,
    size: tuple,
    mode: str = "bicubic",
) -> torch.Tensor:
    """Interpolate a 2-D complex tensor by splitting into mag/phasor channels."""
    mag = A.abs()
    phasor = A / (mag + 1e-12)
    mag_up = _interpolate_real(mag, size=size, mode=mode)
    phasor_up = _interpolate_complex_phasor(phasor, size=size, mode=mode)
    return mag_up * phasor_up / (phasor_up.abs() + 1e-12)

def _softplus_inverse(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Numerically stable inverse of F.softplus."""
    x = x.clamp(min=eps)
    return x + torch.log(-torch.expm1(-x))  # = log(exp(x) - 1), stable for all x > 0

# =============================================================================
# LUTModule - grayscale → phase look-up table + optional depth map
# =============================================================================

class LUTModule(PhysicalModule):
    """
    Learnable monotone look-up table: normalised grayscale in [0, 1] → phase.

    The LUT is parameterised by per-bin slopes (squared to enforce positivity),
    integrated with cumsum to guarantee monotonicity.  An optional per-pixel
    depth map scales the output phase spatially.

    Parameters
    ----------
    geometry    : OpticsGeometry
    n_bins      : number of LUT bins (resolution of the piecewise-linear map)
    use_depth   : if True, adds a per-pixel depth map as a learnable parameter
    init_scale  : total phase span that the initial linear LUT spans (radians)
    """

    def __init__(
        self,
        geometry: OpticsGeometry,
        n_bins: int      = 10,
        use_depth: bool  = False,
        depth_grid_divisor: int = 20,
        init_scale: float = 2 * np.pi,
        init_depth_map: Optional[torch.Tensor] = None,
    ):
        super().__init__(geometry)

        # --- learnable parameters ---

        self._n_bins     = n_bins
        self.phase_slopes_raw = nn.Parameter(
                torch.zeros(n_bins, dtype=torch.float, device=geometry.device)
            )
        self.set_phase_slopes(init_scale)

        self._use_depth  = use_depth
        if use_depth:
            H, W = geometry.slm_pixels
            d = depth_grid_divisor
            assert H % d == 0 and W % d == 0, "slm_pixels must be divisible by depth_grid_divisor"
            self._depth_grid_shape = (H // d, W // d)

            # default: raw = 0 -> depth = 1 everywhere (identity, avoids bootstrapping deadlock)
            self.depth_raw = nn.Parameter(
                torch.zeros(self._depth_grid_shape, dtype=torch.float, device=geometry.device)
            )
            
            self._depth_init_offset = _softplus_inverse(torch.tensor(1.0))
            if init_depth_map is not None:
                self.depth_map = init_depth_map  # goes through the validated setter

    # --- property interface ---

    @property
    def phase_nodes(self) -> torch.Tensor:
        """Return the LUT phase nodes (cumulative integral of slopes)."""
        if self.is_frozen:
            return self._phase_nodes_cache
        else:
            return self._build_phase_nodes()
        
    @phase_nodes.setter
    def phase_nodes(self, value: torch.Tensor):
        """Set the LUT phase nodes directly (overwrites slopes)."""
        if value.shape != (self._n_bins + 1,):
            raise ValueError(f"phase_nodes must have shape ({self._n_bins + 1},)")
        with torch.no_grad():
            slopes = torch.diff(value)
            slopes = torch.clamp(slopes, min=1e-6)
            self.phase_slopes_raw.copy_(torch.sqrt(slopes))
        
    def set_phase_slopes(self, scale: float):
        """Set the phase LUT slopes by providing the overall phase scale."""
        if scale <= 0:
            raise ValueError("Phase scale must be positive.")
        with torch.no_grad():
            slopes_raw = torch.sqrt(torch.full((self._n_bins,), scale, device=self.geometry.device))
            self.phase_slopes_raw.copy_(slopes_raw)

    def phase(self, g: torch.Tensor) -> torch.Tensor:
        """Return the LUT phase values by interpolating the phase nodes."""
        phase_nodes = self.phase_nodes
        g_s = g * self._n_bins
        idx   = torch.clamp(g_s.long(), 0, self._n_bins - 1)
        frac  = g_s - idx
        phase = phase_nodes[idx] + frac * (phase_nodes[idx + 1] - phase_nodes[idx])
        return phase

    @property
    def use_depth(self) -> bool:
        return self._use_depth

    @property
    def depth_map(self) -> torch.Tensor:
        if not self._use_depth:
            return torch.ones(self.geometry.slm_pixels, dtype=torch.float, device=self.geometry.device)
        
        if self.is_frozen:
            return self._depth_map_cache
        else:
            return self._build_depth_map()

    @depth_map.setter
    def depth_map(self, value: torch.Tensor):
        """
        Set the depth map from a physical (x, y) map of relative LC-layer
        thickness, d(x,y)/d0. Any input shape is accepted: if it doesn't
        match the coarse grid, it is bilinearly resized onto it first.

        The input is renormalised to mean 1 before storage (see gauge note
        in _build_depth_map).
        """
        if not self._use_depth:
            raise ValueError("Depth map is not enabled for this LUTModule.")

        value = value.to(dtype=torch.float, device=self.geometry.device)

        if tuple(value.shape) != tuple(self._depth_grid_shape):
            value = _interpolate_real(value, size=self._depth_grid_shape, mode="bilinear")

        with torch.no_grad():
            # enforce the same mean=1 gauge as _build_depth_map, so the
            # stored raw parameters are consistent with what forward() will renormalise to
            self.depth_raw.copy_(_softplus_inverse(value.clamp(min=1e-3)) - self._depth_init_offset)

    # --- PhysicalModule interface ---

    def _build_cached_buffers(self):
        self.register_buffer(
            "_phase_nodes_cache",
            self._build_phase_nodes().detach(),
            persistent=False
        )
        if self._use_depth:
            self.register_buffer(
                "_depth_map_cache", 
                self._build_depth_map().detach(), 
                persistent=False)

    def _clear_cached_buffers(self):
        if "_phase_nodes_cache" in self._buffers:
            del self._buffers["_phase_nodes_cache"]
        if "_depth_map_cache" in self._buffers:
            del self._buffers["_depth_map_cache"]

    def _build_phase_nodes(self):
        """Rebuild the phase nodes from the current slopes."""
        slopes = self.phase_slopes_raw ** 2
        dx     = 1.0 / self._n_bins
        u      = torch.cumsum(slopes * dx, dim=0)
        phase_nodes = torch.cat([torch.zeros(1, device=u.device), u])
        return phase_nodes

    def _build_depth_map(self) -> torch.Tensor:
        EPS = 1e-8
        coarse = F.softplus(self.depth_raw + self._depth_init_offset)
        full = _interpolate_real(coarse, size=self.geometry.slm_pixels, mode="bilinear")
        depth = full / (full.mean() + EPS)
        return depth

    def _forward_disabled(self, grayscale: torch.Tensor) -> torch.Tensor:
        """Linear phase ramp: φ = g · 2π.  No LUT shaping, no depth map."""
        return grayscale * (2 * torch.pi)

    def _forward_enabled(self, grayscale: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        grayscale : (..., H, W) float in [0, 1]

        Returns
        -------
        phase : (..., H, W) float in radians
        """
        phase = self.phase(grayscale)
        if self._use_depth:
            phase = phase * self.depth_map
        return phase


# =============================================================================
# PixelModule - pixel crosstalk and far-field envelope
# =============================================================================

class PixelModule(PhysicalModule):
    """
    SLM pixel crosstalk: each pixel's phase bleeds into neighbours via a
    learnable Gaussian kernel + optional residual. Also, computes the far-field envelope.

    The forward path upsamples φ by K (sub-pixel resolution), convolves with
    the kernel, then computes the equivalent near-field E on the SLM sampling
    grid via either a fast FFT-crop approximation or a full CZT.

    The far-field is extended by a margin (for PupilModule's OTF tile convolution)

    Parameters
    ----------
    init_sigma    : initial Gaussian σ in pixel units (y, x)
    pixel_samples : sub-pixel oversampling factor K
    kernel_pixels : kernel half-width in pixels
    use_residual  : learnable residual on top of the Gaussian
    init_deadspace_reflectance : initial complex deadspace reflectance,
        relative to unit electrode reflectance.
    fast_approx   : use FFT-crop (True) or full CZT (False)
    pupil_num_tiles : must match PupilModule.num_tiles -- see above.
    """

    def __init__(
        self,
        geometry:      OpticsGeometry,
        init_sigma:    tuple = (0.3, 0.3),
        pixel_samples: int   = 3,
        kernel_pixels: int   = 3,
        use_residual:  bool  = True,
        init_deadspace_reflectance: float = 0.8,
        fast_approx:   bool  = True,
        pupil_num_tiles: Optional[int] = None,
        fft_chunk_bytes: int = 128 << 20,
    ):
        self._pixel_samples = pixel_samples
        self._kernel_pixels = kernel_pixels
        self._use_residual  = use_residual
        self._fast_approx   = fast_approx
        self._kernel_samples = kernel_pixels * pixel_samples
        self._pupil_num_tiles = pupil_num_tiles
        self.fft_chunk_bytes = fft_chunk_bytes

        super().__init__(geometry)

        self.sigma_raw = nn.Parameter(torch.empty(2, dtype=torch.float, device=geometry.device))
        self.sigma = init_sigma

        if use_residual:
            self.residual = nn.Parameter(
                torch.zeros(self._kernel_samples, self._kernel_samples, device=geometry.device)
            )

        # Deadspace reflectance, real in (-1, 1). The fill factor is not fitted:
        # it comes from the electrode width in the geometry.
        self.deadspace_reflectance_raw = nn.Parameter(torch.empty(1, dtype=torch.float, device=geometry.device))
        self.deadspace_reflectance = init_deadspace_reflectance

    # --- property interface ---
    @property
    def sigma(self) -> torch.Tensor:
        """Return σ as physical pixel values (y, x)."""
        return F.softplus(self.sigma_raw)

    @sigma.setter
    def sigma(self, value):
        """Set σ from physical pixel values (y, x)."""
        EPS = 1e-6
        v = torch.tensor(value, dtype=torch.float, device=self.geometry.device)
        with torch.no_grad():
            self.sigma_raw.copy_(_softplus_inverse(v, eps=EPS))

    @property
    def electrode_width(self) -> float:
        """Electrode width fx = fy in pixel-pitch units, from the geometry."""
        return self.geometry.electrode_width_norm

    @property
    def fill(self) -> float:
        """Pixel fill factor F = fx*fy, from the geometry. Not a fitted quantity."""
        return self.geometry.fill_factor

    @property
    def deadspace_reflectance(self) -> torch.Tensor:
        """
        Deadspace reflectance rho, relative to unit electrode reflectance.
        Real, confined to (-1, 1) by a tanh: the far-field intensity is
        insensitive to Im(rho) to first order, while the sign is identifiable
        and physical (rho < 0 is a pi phase from electrode-vs-trench height).
        """
        return torch.tanh(self.deadspace_reflectance_raw)

    @deadspace_reflectance.setter
    def deadspace_reflectance(self, value):
        """
        Set the deadspace reflectance. |rho| = 1 is at infinity under tanh, and
        is also the degenerate point at which the fill factor stops affecting
        the envelope, so initialise around 0.7-0.9.
        """
        EPS = 1e-6
        v = torch.as_tensor(value, device=self.geometry.device).detach()
        if v.is_complex():
            if v.imag.abs() > 1e-6:
                warnings.warn(
                    f"Deadspace reflectance has imaginary part {v.imag.item():+.4g}, "
                    f"which is not modelled; keeping the real part only."
                )
            v = v.real
        v = v.to(torch.float)
        if v.abs() > 1.0 + EPS:
            warnings.warn(f"Deadspace reflectance {v.item():+.4f} outside (-1, 1); clamping.")
        with torch.no_grad():
            self.deadspace_reflectance_raw.copy_(torch.atanh(v.clamp(-1.0 + EPS, 1.0 - EPS)))

    @property
    def kernel(self) -> torch.Tensor:
        """Return the Gaussian kernel (with optional residual)."""
        if self.is_frozen:
            return self._kernel_cache
        else:
            return self._build_kernel()
        
    @property
    def aperture_envelope(self) -> torch.Tensor:
        """
        (Mf, Nf) physical pixel envelope: the normalised transform of the
        electrode-deadspace structure alone, without any sampling correction.
        """
        return self._build_aperture_envelope(extended=False)

    @property
    def envelope(self) -> torch.Tensor:
        """
        (Mf, Nf) far-field envelope H[m,n]: the sub-pixel sampling cell together
        with the physical pixel aperture, normalised to unity at the origin.
        """
        return self._get_envelope(extended=False)

    @property
    def fast_approx(self) -> bool:
        """Return whether the fast FFT-crop approximation is used."""
        return self._fast_approx
    
    @fast_approx.setter
    def fast_approx(self, value: bool):
        """Set whether to use the fast FFT-crop approximation."""
        if not isinstance(value, bool):
            raise ValueError("fast_approx must be a boolean.")
        self._fast_approx = value
        self.rebuild()  # rebuild buffers if approximation method changes

    @property
    def far_fov_ext_samples(self) -> tuple:
        """
        (Mf_ext, Nf_ext): the far-field shape this module actually returns
        from forward() -- the nominal (Mf, Nf) FOV plus the extra half-tile
        margin on each side that PupilModule's OTF tile convolution needs
        (see class docstring). Equals geometry.far_fov_samples exactly when
        pupil_num_tiles was not given at construction.
        """
        return (self._Mf_ext, self._Nf_ext)

    # --- PhysicalModule interface ---
 
    def _build_static_buffers(self):
        g   = self.geometry
        K   = self._pixel_samples
        Ks  = self._kernel_samples
        Kp  = self._kernel_pixels
 
        # coordinate grids for the Gaussian kernel
        x = torch.linspace(-Kp / 2, Kp / 2, Ks, device=g.device)
        Ygrid, Xgrid = torch.meshgrid(x, x, indexing="ij")
        self.register_buffer("_grid_X", Xgrid.contiguous().clone())
        self.register_buffer("_grid_Y", Ygrid.contiguous().clone())
 
        # size bookkeeping used by all path variants
        self._K      = K
        self._Ks     = Ks
        self._PK     = g.P * K
        self._QK     = g.Q * K
        self._MK     = g.M * K
        self._NK     = g.N * K
        
        # Lag of the kernel's first tap relative to the output sample. See _build_kbar.
        self._conv_crop = (Ks - 1) // 2

        # extended far-field margin for PupilModule's OTF tile convolution
        # (see class docstring) - must match PupilModule's own Fy/Fx, which
        # is why pupil_num_tiles has to be the SAME value passed to both.
        if self._pupil_num_tiles is not None:
            nt = self._pupil_num_tiles
            if g.Mf % nt != 0 or g.Nf % nt != 0:
                warnings.warn(
                    f"Far-field size ({g.Mf},{g.Nf}) is not divisible by "
                    f"pupil_num_tiles={nt}; the extended far-field margin "
                    f"computed here may not exactly match PupilModule's tile stride."
                )
            self._margin_y = g.Mf // nt
            self._margin_x = g.Nf // nt
        else:
            self._margin_y = 0
            self._margin_x = 0
        self._Mf_ext = g.Mf + 2 * self._margin_y
        self._Nf_ext = g.Nf + 2 * self._margin_x

        # SLM-plane sample counts of the band-limited equivalent field used by
        # the fast path (_get_U_fast).
        self._Pf_ext = max(2, 2 * int(round(self._Mf_ext * g.P / g.M / 2)))
        self._Qf_ext = max(2, 2 * int(round(self._Nf_ext * g.Q / g.N / 2)))

        # Pf_ext has to be an integer (and, for clean centring, an even one)
        for name, got, want in (("rows", self._Pf_ext, self._Mf_ext * g.P / g.M),
                                ("cols", self._Qf_ext, self._Nf_ext * g.Q / g.N)):
            if abs(got - want) / want > 1e-3:
                warnings.warn(
                    f"Fast-path SLM-plane crop along {name} rounds to {got} where "
                    f"{want:.2f} is needed, a {100 * abs(got - want) / want:.2f}% "
                    f"scale error in the reconstruction. Use fast_approx=False, or "
                    f"pick fov/num_tiles so that (Mf + 2*Mf//num_tiles)*P/M is an "
                    f"even integer."
                )

        # Far-field frequency axes, and the parts of the envelope that depend
        # only on the geometry. 
        for suffix, extended in (("_ext", True), ("", False)):
            v, u = self._get_envelope_axes(extended=extended)
            self.register_buffer(f"_v{suffix}", v, persistent=False)
            self.register_buffer(f"_u{suffix}", u, persistent=False)
            # ideal fully-filled square pixel, sinc(u)sinc(v)
            self.register_buffer(f"_sinc_v{suffix}", torch.sinc(v), persistent=False)
            self.register_buffer(f"_sinc_u{suffix}", torch.sinc(u), persistent=False)
            # sub-pixel sampling cell, sinc(u/K)sinc(v/K)
            self.register_buffer(f"_sincK_v{suffix}", torch.sinc(v / K), persistent=False)
            self.register_buffer(f"_sincK_u{suffix}", torch.sinc(u / K), persistent=False)

        if max(float(self._u_ext.abs().max()), float(self._v_ext.abs().max())) > 0.9:
            warnings.warn(
                "The far-field grid reaches |u| > 0.9, where sinc(u)sinc(v) approaches "
                "its zero at the diffraction-order boundary. The pixel envelope divides "
                "by it and will become ill-conditioned there; reduce fov."
            )

    def _build_cached_buffers(self):
        self.register_buffer("_kernel_cache", self._build_kernel().detach())
        self.register_buffer("_kbar_cache", self._build_kbar().detach())
        self.register_buffer("_envelope_cache",
                             self._build_envelope(extended=False).detach(), persistent=False)
        self.register_buffer("_envelope_ext_cache",
                             self._build_envelope(extended=True).detach(), persistent=False)

    def _clear_cached_buffers(self):
        for name in ("_kernel_cache", "_kbar_cache",
                     "_envelope_cache", "_envelope_ext_cache",
                     "_kernel_ft_cache"):  # legacy name, harmless if absent
            if name in self._buffers:
                del self._buffers[name]

    def _build_kernel(self) -> torch.Tensor:
        """
        Build the normalised kernel *with* gradient tracking.
        Called inside _get_kernel_ft every forward pass during training.
        During frozen inference _get_kernel_ft uses the detached self._kernel.
        """
        EPS   = 1e-6
        sigma = self.sigma
        k     = torch.exp(
            -0.5 * ((self._grid_X / sigma[1]) ** 2 + (self._grid_Y / sigma[0]) ** 2)
        )
        if self._use_residual and hasattr(self, "residual"):
            k = (k + self.residual).abs()
        return k / (k.sum() + EPS)
    
    def _get_envelope_axes(self, extended: bool = False) -> tuple:
        """
        Far-field frequency axes (v, u), normalised so that the first
        diffraction order spans [-0.5, 0.5]; they exceed that if fov > 1.

        `extended` selects the (Mf_ext, Nf_ext) shape -- the nominal FOV plus
        the pupil tile margin -- rather than the nominal (Mf, Nf) FOV. Anything
        multiplied onto the far field BEFORE PupilModule crops back down needs
        the extended axes.
        """
        g = self.geometry
        Mf, Nf = (self._Mf_ext, self._Nf_ext) if extended else (g.Mf, g.Nf)
        v = (torch.arange(Mf, dtype=torch.float, device=g.device) - Mf // 2) / g.M
        u = (torch.arange(Nf, dtype=torch.float, device=g.device) - Nf // 2) / g.N
        return v, u

    def _get_envelope_pixel(self, extended: bool = True) -> torch.Tensor:
        """
        Envelope of an ideal, fully reflective square pixel, sinc(u)sinc(v).
        Used by the disabled path, where the far field is computed directly on
        the SLM pixel grid with no sub-pixel sampling.
        """
        sv = self._sinc_v_ext if extended else self._sinc_v
        su = self._sinc_u_ext if extended else self._sinc_u
        return sv.unsqueeze(1) * su.unsqueeze(0)

    def _build_aperture_envelope(self, extended: bool = True) -> torch.Tensor:
        """
        The physical pixel envelope: the normalised transform of the
        electrode-deadspace structure,

            Ahat(u,v) / Ahat(0,0),
            Ahat = (1 - rho) F sinc(fx u) sinc(fy v) + rho sinc(u) sinc(v).

        This is the envelope the pixel structure imposes on the far field,
        independent of how the field is sampled.
        """
        sv, su = (self._sinc_v_ext, self._sinc_u_ext) if extended else (self._sinc_v, self._sinc_u)
        v,  u  = (self._v_ext, self._u_ext) if extended else (self._v, self._u)

        F   = self.fill
        fx  = self.electrode_width
        rho = self.deadspace_reflectance

        ideal = sv.unsqueeze(1) * su.unsqueeze(0)                       # sinc(u)sinc(v)
        elec  = torch.sinc(fx * v).unsqueeze(1) * torch.sinc(fx * u).unsqueeze(0)
        return ((1 - rho) * F * elec + rho * ideal) / (F + (1 - F) * rho)

    def _build_envelope(self, extended: bool = True) -> torch.Tensor:
        """
        The full envelope H(u,v) applied to the far field,

            H = [sinc(u/K) sinc(v/K)] * Ahat_norm(u,v) / [sinc(u) sinc(v)].

        The first factor is the sub-pixel sampling cell. The second replaces the
        aperture: sinc(u)sinc(v) is the envelope of the ideal, fully filled pixel
        implicit in transforming the modulated field alone, and Ahat_norm is the
        physical one that takes its place. H is therefore a correction relative to
        a fully filled pixel, not the pixel envelope itself -- see
        `aperture_envelope` for that.
        """
        sv, su = (self._sinc_v_ext, self._sinc_u_ext) if extended else (self._sinc_v, self._sinc_u)
        kv, ku = (self._sincK_v_ext, self._sincK_u_ext) if extended else (self._sincK_v, self._sincK_u)
        ideal = sv.unsqueeze(1) * su.unsqueeze(0)
        return (kv.unsqueeze(1) * ku.unsqueeze(0)) * self._build_aperture_envelope(extended) / ideal

    def _get_envelope(self, extended: bool = True) -> torch.Tensor:
        if self.is_frozen:
            return self._envelope_ext_cache if extended else self._envelope_cache
        return self._build_envelope(extended=extended)

    # --- internal signal path ---

    def _build_kbar(self) -> torch.Tensor:
        """
        The crosstalk kernel pre-convolved with the K x K sub-pixel box, i.e.
        the effective (Ks + K - 1)^2 stencil that acts directly on the SLM-
        resolution phase map.
        """
        kernel = self._build_kernel()
        K = self._K
        box = torch.ones(1, 1, K, K, dtype=kernel.dtype, device=kernel.device)
        return F.conv_transpose2d(kernel.unsqueeze(0).unsqueeze(0), box)[0, 0]

    def _get_kbar(self) -> torch.Tensor:
        # During training: rebuild with full gradient tracking every call, so
        # sigma_raw and residual receive gradients.
        if self.is_frozen:
            return self._kbar_cache
        return self._build_kbar()

    def _crosstalk_phase(self, phi: torch.Tensor) -> torch.Tensor:
        """Sub-pixel crosstalk-convolved phase on the (PK, QK) grid."""
        kbar = self._get_kbar()
        c    = self._conv_crop
        y = F.conv_transpose2d(
            phi.unsqueeze(1), kbar.unsqueeze(0).unsqueeze(0), stride=self._K
        )
        return y[:, 0, c:c + self._PK, c:c + self._QK]

    def _get_E_upsampled(self, phi: torch.Tensor, E_in: torch.Tensor = None) -> torch.Tensor:
        K   = self._K
        g   = self.geometry
        P, Q = g.slm_pixels

        phi_conv = self._crosstalk_phase(phi)                    # (B, PK, QK)

        if E_in is None:
            E_in = torch.ones(g.slm_pixels, dtype=torch.cfloat, device=phi.device)

        # The nearest-neighbour upsampling of E_in is never materialised: a
        # contiguous (B, P, K, Q, K) tensor reshapes to (B, PK, QK) with exactly
        # the Kronecker index mapping, so E_in broadcasts into place.
        B  = phi_conv.shape[0]
        pc = phi_conv.reshape(B, P, K, Q, K)
        Ei = E_in.reshape(P, 1, Q, 1)

        if E_in.requires_grad:
            # --- training path -------------------------------------------
            # Plain complex product: E_in is never decomposed into modulus and
            # argument on a differentiated path.
            E_up = torch.exp(1j * pc) * (Ei / K)
        else:
            # --- frozen / CGH path ---------------------------------------
            # Fused: one complex allocation for the whole upsampled field
            # rather than two. Only valid when E_in carries no gradient.
            E_up = torch.polar(Ei.abs() / K, pc + torch.angle(Ei))
        return E_up.reshape(B, self._PK, self._QK)

    def _chunks(self, E_up: torch.Tensor, L_row: int, L_col: int, n_row: int) -> tuple:
        """Rows/cols per block for the two 1-D FFT passes, from the byte budget."""
        el = 8 if E_up.dtype == torch.cfloat else 16
        B  = E_up.shape[0]
        return (
            _auto_chunk(E_up.shape[-1], B * L_row * el, self.fft_chunk_bytes),
            _auto_chunk(n_row,          B * L_col * el, self.fft_chunk_bytes),
        )

    def _get_U_fast(self, E_up: torch.Tensor) -> torch.Tensor:
        """
        FFT-crop approximation. Run as two cropped 1-D passes instead of one full fft2.

        The SLM-plane crop widths (Pf_ext, Qf_ext) are tied to the far-field
        sampling by Pf_ext = Mf_ext * P / M -- see _build_static_buffers. They
        are NOT Mf_ext // 2 in general; that only holds for scale == 2 with a
        non-square far field.
        """
        PK, QK = self._PK, self._QK
        Pf_ext, Qf_ext = self._Pf_ext, self._Qf_ext
        c_row, c_col = self._chunks(E_up, PK, QK, Pf_ext)

        U = _centred_fft_axis(E_up, -2, PK, Pf_ext, chunk=c_row)
        U = _centred_fft_axis(U,    -1, QK, Qf_ext, chunk=c_col)
        U = U / (PK * QK) ** 0.5

        E_equiv = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(U), norm="ortho"))
        # extended shape (nominal FOV + pupil tile margin), not g.far_fov_samples
        # directly -- see PixelModule class docstring
        return _fft2_padded(E_equiv, out_shape=self.far_fov_ext_samples)

    def _get_U_full(self, E_up: torch.Tensor) -> torch.Tensor:
        """
        Exact band-limited far field: the DFT of the upsampled SLM field on the
        1/(MK) frequency grid, cropped to the extended FOV.
        """
        MK, NK = self._MK, self._NK
        c_row, c_col = self._chunks(E_up, MK, NK, self._Mf_ext)

        U = _centred_fft_axis(E_up, -2, MK, self._Mf_ext, chunk=c_row)
        U = _centred_fft_axis(U,    -1, NK, self._Nf_ext, chunk=c_col)
        return U / (MK * NK) ** 0.5

    # --- PhysicalModule forward ---

    def _forward_disabled(self, phi: torch.Tensor, E_in: torch.Tensor = None) -> torch.Tensor:
        """Perfect pixels: E = exp(iφ).  No sub-pixel bleed, no kernel.
        Far-field is the ideal FFT of the SLM field, resized (exactly --
        see _resize_circular) to the extended (nominal FOV + pupil tile
        margin) shape; PupilModule crops the margin back off at the end.
        """
        E_ideal = torch.exp(1j * phi)
        if E_in is not None:
            E_ideal = E_ideal * E_in

        U_ideal_first = _fft2_padded(E_ideal, out_shape=self.geometry.far_first_samples)
        U_ideal = _resize_circular(U_ideal_first, self.far_fov_ext_samples)

        return U_ideal * self._get_envelope_pixel()  # correct the numerical envelope due to pixel sampling
 
    def _forward_enabled(self, phi: torch.Tensor, E_in: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        phi : (B, P, Q) float — SLM phase in radians
 
        Returns
        -------
        U : (B, Mf, Nf) complex — equivalent far-field with crosstalk and envelope applied
        """
        K = self._K
        if K == 1 and self._kernel_pixels == 1:
            return self._forward_disabled(phi)
 
        E_up = self._get_E_upsampled(phi, E_in)
        U = self._get_U_fast(E_up) if self._fast_approx else self._get_U_full(E_up)
        U = U * self._get_envelope()   # sub-pixel sampling cell + pixel aperture
        
        return U


# =============================================================================
# SLMFieldModule - effective incident field on the SLM plane
# =============================================================================

class SLMFieldModule(PhysicalModule):
    """
    Learnable incident field on the SLM plane.

    Parameterised as a normalised complex field E (amplitude + phase) and a
    scalar energy scale.  During training E is renormalised each forward pass
    to keep the scale parameter meaningful.  When fov > 1 the field is
    bicubic-interpolated to the expanded SLM sampling grid.

    Parameters
    ----------
    geometry    : OpticsGeometry
    init_field  : complex Tensor of shape slm_pixels — if None, uniform amplitude
    use_scale   : if True, normalise E to unit amplitude and multiply by the learnable scale parameter
    """

    def __init__(
        self,
        geometry:    OpticsGeometry,
        init_field:  Optional[torch.Tensor] = None,
        use_scale:   bool = True,
    ):
        super().__init__(geometry)

        self._shape = geometry.slm_pixels
        self._full_res = True  
        self.E = nn.Parameter(
            torch.ones(self._shape, dtype=torch.cfloat, device=geometry.device)
        )
        
        self._use_scale = use_scale
        if self._use_scale:
            self.E_scale = nn.Parameter(
                torch.tensor(1.0, dtype=torch.float, device=geometry.device)
            )

        if init_field is not None:
            self.set_shape(init_field.shape, retain_field=False)
            self.field = init_field  # goes through the validated setter

    # --- property interface ---

    @property
    def _field_is_passthrough(self) -> bool:
        """
        True when field == self.E exactly, with no rescale and no
        interpolation in between (use_scale=False, running at full SLM
        resolution). In that case there is nothing to cache -- field is
        just self.E itself, not a derived tensor, so a "cache" would only
        be a redundant full-size duplicate of an existing Parameter for
        zero compute saved.
        """
        return (not self._use_scale) and self._full_res

    def _build_field(self) -> torch.Tensor:
        if self._use_scale:
            EPS = 1e-8
            rms = torch.sqrt((self.E.abs() ** 2).mean() + EPS)
            E = self.E / rms * self.E_scale
        else:
            E = self.E

        if self._full_res:
            return E
        else:
            return _interpolate_complex(E, size=self.geometry.slm_pixels)

    @property
    def field(self) -> torch.Tensor:
        """
        Complex field. Shape: slm_pixels.

        If use_scale=False and the field is already at full SLM resolution,
        this is a pure pass-through of self.E (see _field_is_passthrough) --
        always returned directly, uncached, regardless of frozen state.
        """
        if self._field_is_passthrough:
            return self.E
        if self.is_frozen:
            return self._field_cache
        return self._build_field()

    @field.setter
    def field(self, E: torch.Tensor):
        if tuple(E.shape) != tuple(self._shape):
            warnings.warn(
                f"Field shape {tuple(E.shape)} does not match expected shape {self._shape}. Setting shape."
            )
            self.set_shape(E.shape, retain_field=False)

        with torch.no_grad():
            if self._use_scale:
                EPS = 1e-8
                rms = torch.sqrt((E.abs() ** 2).mean() + EPS)
                self.E.copy_((E / rms).to(dtype=torch.cfloat, device=self.geometry.device))
                self.scale = rms.item()  # set the scale parameter to the RMS of the new field
            else:
                self.E.copy_(E.to(dtype=torch.cfloat, device=self.geometry.device))

        if self.is_frozen:
            self._clear_cached_buffers()
            self._build_cached_buffers()

    @property
    def scale(self) -> float:
        """The current scale factor."""
        if self._use_scale:
            return self.E_scale
        else:
            return 1.0
        
    @scale.setter
    def scale(self, value: float):
        if not self._use_scale:
            raise ValueError("Scale parameter is not enabled.")
        with torch.no_grad():
            self.E_scale.fill_(value)

    # --- PhysicalModule interface (caching) ---

    def _build_cached_buffers(self):
        # Passthrough configs (use_scale=False, full res) have nothing
        # worth caching -- field IS self.E, see _field_is_passthrough.
        if not self._field_is_passthrough:
            self.register_buffer("_field_cache", self._build_field().detach())

    def _clear_cached_buffers(self):
        if "_field_cache" in self._buffers:
            del self._buffers["_field_cache"]

    # --- User interface ---

    def set_shape(self, new_shape: tuple, retain_field: bool = True):
        """
        Set the shape of the learnable field. If retain_field is True, the existing field is interpolated to the new shape.
        Otherwise, the field is reset to uniform amplitude.  If the new shape is different from the SLM pixels, 
        the field will be interpolated to the SLM grid during forward passes.

        Parameters
        ----------
        new_shape : tuple of ints — new shape of the learnable field
        retain_field : bool — if True, interpolate the existing field to the new shape; if False, reset to uniform amplitude
        """
        if tuple(new_shape) == tuple(self._shape):
            return  # no change
        
        self._shape = new_shape
        if new_shape != self.geometry.slm_pixels:
            self._full_res = False
        else:
            self._full_res = True
            
        with torch.no_grad():
            if retain_field:
                E_old = self.E.detach().clone()
                E_interp = _interpolate_complex(E_old, size=new_shape)
                self.E = nn.Parameter(E_interp)
            else:
                self.E = nn.Parameter(
                    torch.ones(self._shape, dtype=torch.cfloat, device=self.geometry.device)
                )

        if self.is_frozen:
            self._clear_cached_buffers()
            self._build_cached_buffers()

    def remove_ramp(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Remove any global linear phase from the field E. 
        This is useful for removing tip-tilt components after training.

        Returns
        -------
        sx : float
            The slope of the linear phase in the x direction (radians per pixel).
        sy : float
            The slope of the linear phase in the y direction (radians per pixel).
        """
        sx, sy = self._get_phase_grad()
        ramp = self._make_phasor_ramp(sx, sy)
        with torch.no_grad():
            self.E.mul_(ramp)

        if self.is_frozen:
            self._clear_cached_buffers()
            self._build_cached_buffers()

        return sx, sy
    
    def clean_aperture(
        self,
        strength: float = 0.25,
        min_dark_area_frac: float = 0.02,
        max_dark_bright_ratio: float = 0.3,
        blur_sigma: float = 15.0,
        close_kernel: int = 15,
    ):
        """
        Remove weak field outside the main illuminated aperture.

        The aperture is estimated automatically from the field amplitude using
        Otsu thresholding. The largest connected bright component is retained,
        morphologically closed, softly blurred, and used as an apodised mask.

        The operation is skipped if no meaningful aperture split is detected
        (e.g. a nearly uniform field or a very small masked region).

        Parameters
        ----------
        strength : float
            Exponent applied to the amplitude before thresholding.  Values < 1
            compress the dynamic range and make the thresholding more robust.
        min_dark_area_frac : float
            Minimum fraction of the field area that must be dark to consider
            the aperture split meaningful.  If the dark area is smaller than
            this fraction, the operation is skipped.
        max_dark_bright_ratio : float
            Maximum ratio of the median dark amplitude to the median bright amplitude.  If the dark region is too bright, the operation is skipped.
        blur_sigma : float
            Standard deviation of the Gaussian blur applied to the aperture mask.
        close_kernel : int 
            Size of the square kernel used for morphological closing of the aperture mask.
        """
        import cv2

        amp = self.field.abs()
        amp_np = amp.detach().cpu().numpy().astype(np.float32)

        # Slight compression improves threshold robustness.
        img = np.power(amp_np, strength)
        img8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        _, mask = cv2.threshold(
            img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        dark_mask = mask == 0
        frac_dark = dark_mask.mean()

        # No meaningful aperture present.
        if frac_dark < min_dark_area_frac or frac_dark > 1.0 - min_dark_area_frac:
            return

        dark_vals = amp_np[dark_mask]
        bright_vals = amp_np[~dark_mask]

        if np.median(dark_vals) > max_dark_bright_ratio * np.median(bright_vals):
            return

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

        if num_labels <= 1:
            return

        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        aperture_mask = labels == largest_label

        kernel = np.ones((close_kernel, close_kernel), np.uint8)
        aperture_mask = cv2.morphologyEx(
            aperture_mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
        ).astype(bool)

        soft_mask = cv2.GaussianBlur(
            aperture_mask.astype(np.float32),
            (0, 0),
            sigmaX=blur_sigma,
        )

        soft_mask = torch.from_numpy(soft_mask).to(
            device=self.geometry.device,
            dtype=amp.dtype,
        )
        
        aperture_mask = torch.from_numpy(aperture_mask).to(
            device=self.geometry.device
        )

        field = self.field * soft_mask
        field[~aperture_mask] = 0

        self.field = field

    # --- Helper methods ---

    def _get_phase_grad(self) -> torch.Tensor:
        """
        Compute the average phase gradient of the field E in the x and y directions.

        Returns
        -------
        sy : float
            The slope of the linear phase in the y direction (radians per pixel).
        sx : float
            The slope of the linear phase in the x direction (radians per pixel).

        Explanation
        -----------
        A phasor E[p,q] = exp(2pi j (u0 p + v0 q)) shifts the far-field by (u0, v0), 
        in the normalised frequency coordinates (u, v in [-0.5, 0.5] for the first order 
        of the far-field). The linear phase is removed by multiplying E by the opposite phasor.

        The computed gradients (sx, sy) are the slopes of the linear phase in radians per pixel,
        related to the normalised frequency by (sx, sy) = 2pi (u0, v0).
        """
        # Normalise the field to unit amplitude to avoid bias from amplitude variations
        E_phasor = self.E / (self.E.abs() + 1e-8)

        # Compute the average phase difference between adjacent pixels in y and x directions
        gy = torch.mean(E_phasor[1:, :] * E_phasor[:-1, :].conj())
        gx = torch.mean(E_phasor[:, 1:] * E_phasor[:, :-1].conj())
        sy = torch.angle(gy.sum())
        sx = torch.angle(gx.sum())

        return sx, sy
    
    def _make_phasor_ramp(self, sx: float, sy: float) -> torch.Tensor:
        """
        Make a complex phasor ramp that, when multiplied with E, removes the linear phase.
        The slopes sx, sy are in radians per pixel.

        Parameters
        ----------
        sx : float
            The slope of the linear phase in the x direction (radians per pixel).
        sy : float
            The slope of the linear phase in the y direction (radians per pixel).

        Returns
        -------
        ramp : torch.Tensor, shape (H, W) complex
            The complex phase ramp that, when multiplied with E, removes the linear phase.
        """
        H, W = self._shape
        y = torch.arange(H, dtype=torch.float, device=self.geometry.device) - H // 2
        x = torch.arange(W, dtype=torch.float, device=self.geometry.device) - W // 2
        Y, X = torch.meshgrid(y, x, indexing='ij')
        ramp = torch.exp(-1j * (sx * X + sy * Y))
        return ramp

    # --- PhysicalModule interface ---
    def _forward_disabled(self) -> torch.Tensor:
        """
        Uniform unit-amplitude plane wave: returns a ones field on the
        (possibly FOV-expanded) SLM grid.  No amplitude or phase shaping.
        """
        return torch.ones(
            self.geometry.slm_samples, dtype=torch.cfloat, device=self.geometry.device
        )

    def _forward_enabled(self) -> torch.Tensor:
        """
        Returns
        -------
        E_inc : (P, Q) complex — incident field on the SLM pixel grid
        """
        E_inc = self.field
        return E_inc


# =============================================================================
# PupilModule - pupil aberrations and vignetting
# =============================================================================

class PupilModule(PhysicalModule):
    """
    Pupil aberrations and vignetting.

    The pupil aberrations are parameterised as a Seidel polynomial with 6 learnable coefficients.  
    The vignetting aperture is optional and is parameterised by the distance of the pupil aperture from the SLM 
    and its (y, x) offset. Aberrations contribute as phase, vignetting contributes as amplitude.  
    They are used to estimate the local OTF at each far-field tile, which is then used to modulate the far-field U.

    The far-field is divided into (num_tiles x num_tiles) overlapping
    tiles, each multiplied by the local aberration OTF sampled at its centre,
    and the result is stitched back together with bilinear weights.

    Parameters
    ----------
    geometry       : OpticsGeometry
    init_coeffs    : SeidelCoefficients — initial values (default all-zero)
    num_tiles      : number of OTF tiles along each far-field dimension
    use_vignetting : if True, apply a near-field vignetting aperture
    init_aperture_offset   : initial (y, x) offset of the vignetting aperture in mm
    init_aperture_distance : initial distance of the vignetting aperture from the SLM
    """

    def __init__(
        self,
        geometry:      OpticsGeometry,
        init_coeffs:   AberrCoefficients = None,
        num_tiles: int  = 40,
        use_vignetting: bool = False,
        init_aperture_offset: tuple = (0.0, 0.0),
        init_aperture_distance: float = None,
        tile_chunk_bytes: int = 128 << 20,
        cache_otf: bool = False,
    ):
        self._num_tiles  = num_tiles
        self._use_vignetting = use_vignetting
        # Peak workspace budget for one block of tile rows in _forward_enabled.
        self.tile_chunk_bytes = tile_chunk_bytes
        # Materialise the whole OTF on freeze()?  At 1920x1200 / num_tiles=40
        # that tensor is (41, 41, 120, 192) complex64 = 310 MB resident, and
        # ~2 GB transient while it is built.  It is cheap to rebuild per tile
        # block from its 8-term separable form (see _build_otf_rows), so the
        # default is now to not hold it.  Set True if enough VRAM is available.
        self._cache_otf = cache_otf
        super().__init__(geometry)

        self.coeffs = nn.Parameter(
            torch.zeros(AberrCoefficients.NUM_COEFFS, dtype=torch.float, device=geometry.device)
        )
        
        if init_coeffs is not None:
            self.seidel_coeffs = init_coeffs

        if use_vignetting:
            self._aperture_softness = 50.0
            self.aperture_offset_raw = nn.Parameter(
                torch.zeros(2, dtype=torch.float, device=geometry.device)
            )
            self.aperture_distance_raw = nn.Parameter(
                torch.ones(1, dtype=torch.float, device=geometry.device)
            )
            self.aperture_offset = init_aperture_offset
            if init_aperture_distance is not None:
                self.aperture_distance = init_aperture_distance 

    # --- property interface ---

    @property
    def seidel_coeffs(self) -> AberrCoefficients:
        return AberrCoefficients.from_tensor(self.coeffs.detach())

    @seidel_coeffs.setter
    def seidel_coeffs(self, c: AberrCoefficients):
        with torch.no_grad():
            self.coeffs.copy_(c.to_tensor(device=self.geometry.device))

    @property
    def otf(self) -> torch.Tensor:
        """
        Return the OTF approximation map for the far-field tiles.
        Shape: (num_nodes_far, num_nodes_far, Ty, Tx) where num_nodes_far 
        is num_tiles + 1, and Ty, Tx are the tile sizes in the far-field.
        """
        if self._otf_is_cached:
            return self._otf_cache
        return self._build_otf()

    @property
    def aperture_offset(self) -> torch.Tensor:
        """
        Return the aperture offset in mm.
        It is constrained to be within the lens diameter.
        """
        if not self._use_vignetting:
            raise AttributeError("Aperture offset is only available when vignetting is enabled.")
        else:
            diameter_mm = self.geometry.lens_diameter * self.geometry.MM_PER_INCH
            return torch.tanh(self.aperture_offset_raw) * diameter_mm
        
    @aperture_offset.setter
    def aperture_offset(self, value):
        """
        Set the aperture offset (y, x) in mm.
        It is constrained to be within the lens diameter.
        """
        if not self._use_vignetting:
            raise AttributeError("Aperture offset is only available when vignetting is enabled.")
        else:
            v = torch.tensor(value, dtype=torch.float, device=self.geometry.device)
            diameter_mm = self.geometry.lens_diameter * self.geometry.MM_PER_INCH

            if torch.any(torch.abs(v) >= diameter_mm):
                raise ValueError(f"Aperture offset {v} exceeds lens diameter {diameter_mm}.")
            
            with torch.no_grad():
                self.aperture_offset_raw.copy_(torch.atanh(v / diameter_mm))

    @property
    def aperture_distance(self) -> torch.Tensor:
        """
        Return the aperture distance in mm.
        It is constrained to be positive and is scaled by the focal length of the system.
        """
        if not self._use_vignetting:
            raise AttributeError("Aperture distance is only available when vignetting is enabled.")
        else:
            focal_length = self.geometry.focal_length
            return F.softplus(self.aperture_distance_raw) * focal_length
    
    @aperture_distance.setter
    def aperture_distance(self, value: float):
        """
        Set the aperture distance from a physical value.
        It is constrained to be positive and is scaled by the focal length of the system.
        """
        if not self._use_vignetting:
            raise AttributeError("Aperture distance is only available when vignetting is enabled.")
        else:
            EPS = 1e-6
            v = torch.tensor(value, dtype=torch.float, device=self.geometry.device)
            with torch.no_grad():
                focal_length = self.geometry.focal_length
                self.aperture_distance_raw.copy_(_softplus_inverse(v / focal_length, eps=EPS))

    @property
    def num_tiles(self) -> int:
        """
        Return the number of OTF tiles along each far-field dimension.
        Must be >= 1. Default is 40.
        """
        return self._num_tiles
    
    @num_tiles.setter
    def num_tiles(self, value: int):
        if value < 1:
            raise ValueError("Number of tiles must be >= 1.")
        if value != self._num_tiles:
            self._num_tiles = value
            self.rebuild()

    # --- PhysicalModule interface ---

    def _build_static_buffers(self):
        g  = self.geometry
        Mf = g.Mf
        Nf = g.Nf
        nt = self._num_tiles

        self._num_nodes_far = nt + 1

        # check if division is exact, otherwise raise a warning
        if Mf % nt != 0:
            warnings.warn(
                f"Number of far-field rows {Mf} is not divisible by number of tiles {nt}. "
                f"This may result in an error in the OTF approximation."
            )
        if Nf % nt != 0:
            warnings.warn(
                f"Number of far-field columns {Nf} is not divisible by number of tiles {nt}. "
                f"This may result in an error in the OTF approximation."
            )

        # tile strides in the far-field
        self._Fy = Mf // nt
        self._Fx = Nf // nt

        # tile size (2x stride to get 50 % overlap)
        Ty, Tx = self._Fy * 2, self._Fx * 2
        self._Ty = Ty
        self._Tx = Tx

        # bilinear stitch weight map — shape (1, 1, 1, Ty, Tx)
        wy = torch.linspace(0, 1, steps=self._Fy, device=g.device)
        wx = torch.linspace(0, 1, steps=self._Fx, device=g.device)
        wm = torch.outer(wy, wx)
        wm = torch.cat([wm, wm.flip(0)], dim=0)
        wm = torch.cat([wm, wm.flip(1)], dim=1)
        self.register_buffer("_weight_map", wm.unsqueeze(0).unsqueeze(0).unsqueeze(0))

    def _build_cached_buffers(self):
        if self._cache_otf:
            self.register_buffer("_otf_cache", self._build_otf().detach(),
                                 persistent=False)

    def _clear_cached_buffers(self):
        if "_otf_cache" in self._buffers:
            del self._buffers["_otf_cache"]

    @property
    def _otf_is_cached(self) -> bool:
        return "_otf_cache" in self._buffers

    def _build_otf(self) -> torch.Tensor:
        """Full OTF, shape (n_far, n_far, Ty, Tx).  Unshifted (public form)."""
        return self._build_otf_rows(0, self._num_nodes_far, shifted=False)

    def _build_otf_rows(self, i0: int, i1: int, shifted: bool = False) -> torch.Tensor:
        """
        OTF for the tile-node rows [i0, i1).

        `shifted=True` returns it already ifftshifted over the last two
        (pupil) axes, which is what _forward_enabled consumes -- see there for
        why that lets four fftshifts on the full 5-D tile tensor collapse into
        this one, free, roll of the 1-D pupil coordinates.
        """
        Y, X, V, U = self._get_yxvu(self._Ty, self._Tx, self._num_nodes_far,
                                    roll_pupil=shifted)
        V, U = V[i0:i1], U[i0:i1]

        otf = self._get_aberration(Y, X, V, U)

        if self._use_vignetting:
            aperture = self._get_aperture(Y, X, V, U)
            otf = otf * aperture

        return otf

    def _get_otf_rows(self, i0: int, i1: int) -> torch.Tensor:
        """ifftshifted OTF rows, from the cache if one is held."""
        if self._otf_is_cached:
            return torch.fft.ifftshift(self._otf_cache[i0:i1], dim=(-2, -1))
        return self._build_otf_rows(i0, i1, shifted=True)

    # --- coordinate grids ---

    def _get_yxvu(self, y_samples: int, x_samples: int, n_far: int,
                  roll_pupil: bool = False):
        """
        Pupil (Y,X): radius 1 at the SLM's true corner (fixed physical aperture,
        P x Q), sampled over the FULL padded/oversampled canvas extent
        (M x N pixel pitches) at (y_samples,x_samples) points.

        Field (V,U): radius 1 at the corner of the first diffraction order,
        tied to (M,N) -- fixed regardless of fov (see earlier fix).
        """
        g = self.geometry
        P, Q = g.P, g.Q

        # radius-1 reference: TRUE (unpadded) SLM corner circle
        orig_diag  = torch.sqrt(torch.tensor(P**2 + Q**2, dtype=torch.float, device=g.device))
        norm_coeff = 2.0 / orig_diag

        # Physical extent actually sampled by one tile's inverse FFT.
        padded_P = g.M
        padded_Q = g.N

        y_fractional = (torch.arange(y_samples, dtype=torch.float, device=g.device) + 0.5) / y_samples - 0.5
        x_fractional = (torch.arange(x_samples, dtype=torch.float, device=g.device) + 0.5) / x_samples - 0.5

        y_norm = y_fractional * padded_P * norm_coeff
        x_norm = x_fractional * padded_Q * norm_coeff
        if roll_pupil:
            # ifftshift of the (Y, X) grid == ifftshift of each 1-D coordinate
            # vector, because every quantity built from them is elementwise.
            y_norm = torch.roll(y_norm, -(y_samples // 2))
            x_norm = torch.roll(x_norm, -(x_samples // 2))
        Y, X = torch.meshgrid(y_norm, x_norm, indexing="ij")

        # field grid: unchanged from the earlier fov-independent fix
        v_1d = torch.linspace(-0.5, 0.5, n_far, device=g.device) * np.sqrt(2.0) * (g.Mf / g.M)
        u_1d = torch.linspace(-0.5, 0.5, n_far, device=g.device) * np.sqrt(2.0) * (g.Nf / g.N)
        V, U = torch.meshgrid(v_1d, u_1d, indexing="ij")

        return Y, X, V, U
    
    # --- aberration and vignetting model ---

    def _get_aberration(self, Y, X, V, U) -> torch.Tensor:
        """
        Compute the Seidel aberration phase for each (u, v) coordinate in the far-field
        as a (x, y) phase map in the SLM plane. The phase is applied as a
        multiplicative factor to the far-field amplitude U = fft(E). Written in a
        separable GEMM form instead of the naive 4-D tensor form to reduce memory usage
        and improve speed.
        
        c: 6 Seidel coefficients (coma, astigmatism, field curvature, distortion, tilt_x, tilt_y)

        Separable form
        --------------
        Substituting Xr = X cos(beta) + Y sin(beta) and expanding, the Seidel
        phase is exactly an 8-term sum of products of a field-only coefficient
        and a pupil-only basis function:

            phase[i,j,y,x] = sum_b  A_b[i,j] * Bs_b[y,x]

        with Bs = (R2*X, R2*Y, X^2, X*Y, Y^2, R2, X, Y).
        """
        h2   = U ** 2 + V ** 2
        h    = torch.sqrt(h2)
        beta = torch.atan2(V, U)
        cb, sb = torch.cos(beta), torch.sin(beta)
        R2   = X ** 2 + Y ** 2

        c = self.coeffs
        A = torch.stack([                                   # (8, n_rows, n_far)
            c[0] * h * cb,
            c[0] * h * sb,
            c[1] * h2 * cb ** 2,
            2.0 * c[1] * h2 * cb * sb,
            c[1] * h2 * sb ** 2,
            c[2] * h2 + c[4] * U + c[5] * V,
            c[3] * h2 * h * cb,
            c[3] * h2 * h * sb,
        ])
        Bs = torch.stack([                                  # (8, Ty, Tx)
            R2 * X, R2 * Y, X * X, X * Y, Y * Y, R2, X, Y,
        ])
        ny, nf = A.shape[1], A.shape[2]
        Ty, Tx = Bs.shape[-2:]
        phase = (A.reshape(8, -1).transpose(0, 1) @ Bs.reshape(8, -1)).reshape(ny, nf, Ty, Tx)
        return torch.exp(1j * phase)
    
    def _get_physical_units(self):
        """
        Compute the physical units of the SLM plane and the far-field plane,
        used to convert from normalized coordinates to physical coordinates in mm 
        at the SLM and aperture planes.

        Returns
        -------
        units_xy : float — physical units of the SLM plane in mm
        units_uv : float — physical units of the far-field plane in mm
        """
        g = self.geometry

        # In X, Y, radius 1 is at the corner of the cornermost pixel of the SLM
        units_xy = np.sqrt(g.P ** 2 + g.Q ** 2) * 0.5 * g.slm_pixel_pitch * 1e-3 

        # In U, V, radius 1 is at the corner of the first diffraction order in the far-field
        # the diffraction angle at edge of the first diffraction order is given by arcsin(wavelength / pixel_pitch) / 2
        # if radius 1 is at the corner of the first diffraction order, then in the normalised units the edge is at U_edge = 1 / sqrt(2)
        # so U / U_edge = tan(theta) / tan(diff_angle) -> tan(theta) = U / U_edge * tan(diff_angle) = U * sqrt(2) * tan(diff_angle)
        # in the aperture plane, the distance from the optical axis is given by tan(theta) * distance, 
        # where distance is the distance from the SLM to the aperture plane
        diff_angle = np.arcsin(g.wavelength / g.slm_pixel_pitch * 1e-3) * 0.5
        distance = self.aperture_distance
        units_uv = np.sqrt(2) * np.tan(diff_angle) * distance

        return units_xy, units_uv
    
    def _get_aperture(self, Y, X, V, U) -> torch.Tensor:
        """
        Compute the vignetting aperture function for each (u, v) coordinate in the far-field
        as a (x, y) amplitude map in the SLM plane.  The aperture is applied as a 
        multiplicative factor to the far-field amplitude U = fft(E).
        """
        Un = U.unsqueeze(-1).unsqueeze(-1)
        Vn = V.unsqueeze(-1).unsqueeze(-1)
        Xn = X.unsqueeze(0).unsqueeze(0)
        Yn = Y.unsqueeze(0).unsqueeze(0)

        # Get the physical units of the SLM plane and the far-field plane
        units_xy, units_uv = self._get_physical_units()

        # Compute the effective coordinates in the aperture plane of the ray emerging from (x, y) in the SLM plane 
        # for each (u, v) coordinate, in mm.
        offset = self.aperture_offset
        Xp = Xn * units_xy + Un * units_uv - offset[1]
        Yp = Yn * units_xy + Vn * units_uv - offset[0]
        aperture_radius = torch.sqrt(Xp**2 + Yp**2)

        lens_radius = self.geometry.lens_diameter * self.geometry.MM_PER_INCH * 0.5  # convert from inches to mm

        # Compute the signed distance function (SDF) of the aperture edge: positive inside the aperture, negative outside
        # make it unitless by dividing by the lens radius
        aperture_sdf = (lens_radius - aperture_radius) / lens_radius

        # Apply a sigmoid function get the smoothed aperture function, which is 1 inside the aperture and 0 outside.
        aperture = torch.sigmoid(self._aperture_softness * aperture_sdf)

        return aperture
    
    # --- user interface ---
    
    def get_critical_aperture_distance(self) -> float:
        """
        Compute the critical aperture distance at which the aperture starts to vignette the far-field.

        Returns
        -------
        critical_distance : float — critical distance in mm

        Explanation
        -----------
        This happens in the worst case when the corner of the SLM diffracts light at the maximum angle, 
        on the diagonal and the ray just grazes the edge of the lens.
        The critical distance is given by the formula:
            lens_radius = slm_radius + tan(diffraction_angle) * critical_distance * sqrt(2)
        where slm_radius is the radius of the SLM in mm, lens_radius is the radius of the lens in mm, 
        and diffraction_angle is the angle of the first diffraction order.
        """

        g = self.geometry
        slm_radius = np.sqrt(g.P ** 2 + g.Q ** 2) * 0.5 * g.slm_pixel_pitch * 1e-3  # in mm
        lens_radius = g.lens_diameter * g.MM_PER_INCH * 0.5  # in mm
        diffraction_angle = np.arcsin(g.wavelength / g.slm_pixel_pitch * 1e-3) * 0.5
        critical_distance = (lens_radius - slm_radius) / (np.tan(diffraction_angle) * np.sqrt(2))
        return critical_distance

    # --- forward ---

    def _expected_input_shape(self) -> tuple:
        Fy, Fx = self._Fy, self._Fx
        return (self.geometry.Mf + 2 * Fy, self.geometry.Nf + 2 * Fx)

    def _check_input_shape(self, U_ideal: torch.Tensor) -> None:
        expected = self._expected_input_shape()
        got = tuple(U_ideal.shape[-2:])
        if got != expected:
            raise RuntimeError(
                f"PupilModule expected an extended far-field of shape {expected} "
                f"(= geometry.far_fov_samples + 2*(Fy,Fx), Fy,Fx derived from "
                f"num_tiles={self._num_tiles}), got {got}. This almost always means "
                f"PixelModule.pupil_num_tiles does not match PupilModule.num_tiles -- "
                f"they must be constructed with the same value, and kept in sync if "
                f"either is changed after construction (e.g. via the num_tiles setter)."
            )

    def _forward_disabled(self, U_ideal: torch.Tensor) -> torch.Tensor:
        """
        No aberrations: crop the extended far-field (see PixelModule's
        far_fov_ext_samples / class docstring) back down to the nominal FOV.
        Cropping still has to happen here regardless of whether aberration
        correction itself is enabled -- PixelModule always outputs the
        extended shape when pupil_num_tiles was configured, and everything
        downstream (envelope, background, camera) expects the nominal
        (Mf, Nf) shape.
        """
        self._check_input_shape(U_ideal)
        Fy, Fx = self._Fy, self._Fx
        if Fy == 0 and Fx == 0:
            return U_ideal
        return U_ideal[..., Fy:-Fy, Fx:-Fx]

    def _forward_enabled(self, U_ideal: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        U_ideal : (B, Mf+2*Fy, Nf+2*Fx) complex — the TRUE (non-periodic)
            far field, already computed out to the extra half-tile margin
            by PixelModule (see PixelModule.far_fov_ext_samples). 

        Returns
        -------
        U_conv : (B, Mf, Nf) complex — aberrated far-field amplitude,
            cropped back down to the nominal FOV.
        """
        self._check_input_shape(U_ideal)
        Fy, Fx = self._Fy, self._Fx
        Ty, Tx = self._Ty, self._Tx
        nn_far = self._num_nodes_far
        B      = U_ideal.shape[0]

        # Output is an integer number of (Fy, Fx) blocks: out_h = num_tiles*Fy
        # + Ty = (num_tiles + 2)*Fy, and likewise for the columns.
        nb_y = self._num_tiles + 2
        nb_x = self._num_tiles + 2
        acc   = U_ideal.new_zeros(B, nb_y * Fy, nb_x * Fx)
        acc_v = acc.view(B, nb_y, Fy, nb_x, Fx)

        # One block of tile rows at a time
        el   = 8 if U_ideal.dtype == torch.cfloat else 16
        rows = _auto_chunk(nn_far, B * nn_far * Ty * Tx * el, self.tile_chunk_bytes)

        for i0 in range(0, nn_far, rows):
            i1 = min(i0 + rows, nn_far)
            r  = i1 - i0

            slab  = U_ideal[:, i0 * Fy : (i1 - 1) * Fy + Ty]
            tiles = slab.unfold(1, Ty, Fy).unfold(2, Tx, Fx)   # (B, r, nn_far, Ty, Tx)

            # fftshift(fft2(ifftshift( fftshift(ifft2(ifftshift(t))) * otf ))) is
            # identically fft2(ifft2(t) * ifftshift(otf)) for even (Ty, Tx), but
            # the ifftshift(otf) is absorbed into the OTF itself
            # So below is still FT centred in the slm and far-field
            otf_c = self._get_otf_rows(i0, i1)
            conv  = torch.fft.fft2(
                torch.fft.ifft2(tiles, norm="ortho") * otf_c.unsqueeze(0),
                norm="ortho",
            ) * self._weight_map

            # Overlap-add. With 50% overlap each tile is exactly 2x2 blocks of
            # (Fy, Fx), so the stitch is four strided accumulations
            for a in (0, 1):
                for b in (0, 1):
                    q = conv[..., a * Fy:(a + 1) * Fy, b * Fx:(b + 1) * Fx]
                    acc_v[:, i0 + a : i0 + a + r, :, b : b + nn_far, :] += \
                        q.permute(0, 1, 3, 2, 4)

        return acc[..., Fy:-Fy, Fx:-Fx]


# =============================================================================
# BackgroundModule - stray light background field in the far-field plane
# =============================================================================

class BackgroundModule(PhysicalModule):
    """
    Learnable stray-light background field in the far-field plane.

    Represented as a complex field E in the far-field, initialised to very low
    amplitude.  The forward() method simply returns E which is then added to
    the signal field U.

    When *disabled*, forward() returns a zero tensor of the correct shape.

    Parameters
    ----------
    geometry   : OpticsGeometry
    init_field : complex Tensor (Mf, Nf) — initial background; if None, random
    """

    def __init__(
        self,
        geometry:   OpticsGeometry,
        init_field: Optional[torch.Tensor] = None,
    ):
        super().__init__(geometry)

        g = geometry

        self._shape = g.far_fov_samples
        self._full_res = True
        # Far-field sampling (M, N) the stored field is defined at. 
        self._field_MN = (g.M, g.N)

        self.U = nn.Parameter(torch.zeros(self._shape, dtype=torch.cfloat, device=g.device))

        if init_field is not None:
            self.field = init_field  # goes through the validated setter
        else:
            # If no initial field is provided, initialize with a small random complex field
            E_init = torch.randn(g.far_fov_samples, dtype=torch.cfloat, device=g.device)
            E_init = E_init / E_init.abs().max() * 1e-3
            self.field = E_init  # goes through the validated setter

    # --- property interface ---

    @property
    def field(self) -> torch.Tensor:
        if self._full_res:
            return self.U
        else:
            return _interpolate_complex(self.U, size=self.geometry.far_fov_samples)

    @field.setter
    def field(self, U: torch.Tensor, override_shape: bool = False):
        if tuple(U.shape) != tuple(self._shape):
            if not override_shape:
                raise ValueError(
                    f"Expected field shape {self._shape}, got {tuple(U.shape)}"
                )
            else:
                self.set_shape(U.shape, retain_field=False)
                
        with torch.no_grad():
            self.U.copy_(U.to(dtype=torch.cfloat, device=self.geometry.device))

    # --- User interface ---

    def apply_saturation_mask(self, mask: torch.Tensor):
        """
        Set the oversaturation mask for the learnable field. The mask should be a boolean tensor of the same shape as the field,
        where True indicates that the corresponding pixel is oversaturated and should be set to 1+0j.

        Parameters
        ----------
        mask : torch.Tensor — boolean tensor of shape (Mf, Nf) indicating oversaturated pixels
        """
        if tuple(mask.shape) != tuple(self._shape):
            raise ValueError(
                f"Expected mask shape {self._shape}, got {tuple(mask.shape)}"
            )
        
        with torch.no_grad():
            # set to extremely high value to indicate oversaturation
            # this is useful for predicting the camera image, but cant be corrected for becase there is 
            # no phase or amplitude information in the oversaturated pixels.
            self.U[mask] = 100 + 0j

    def set_shape(self, new_shape: tuple, retain_field: bool = True):
        """
        Set the shape of the learnable field. If retain_field is True, the existing field is interpolated to the new shape.
        Otherwise, the field is reset to uniform amplitude.  If the new shape is different from the SLM pixels, 
        the field will be interpolated to the SLM grid during forward passes.

        Parameters
        ----------
        new_shape : tuple of ints — new shape of the learnable field
        retain_field : bool — if True, interpolate the existing field to the new shape; if False, reset to uniform amplitude
        """
        if tuple(new_shape) == tuple(self._shape):
            return  # no change
        
        self._shape = new_shape
        if new_shape != self.geometry.far_fov_samples:
            self._full_res = False
        else:
            self._full_res = True
            
        with torch.no_grad():
            if retain_field:
                U_old = self.U.detach().clone()
                U_interp = _interpolate_complex(U_old, size=new_shape)
                self.U = nn.Parameter(U_interp)
            else:
                self.U = nn.Parameter(
                    torch.zeros(new_shape, dtype=torch.cfloat, device=self.geometry.device)
                )

    def get_filtered(self, mode: str="slm"):
        """
        Return a saturation-thresholded copy of U and its IFFT (SLM-plane field).
        Useful for diagnostics: oversaturated pixels are set to 1+0j.

        Parameters
        ----------
        mode : str — "slm", "far", or "both"; determines which field to return

        Returns
        -------
        E_thres  : (Mf, Nf) complex — thresholded far-field background
        U_masked : (Mf, Nf) complex — SLM-plane field corresponding to the masked far-field background
        """
        with torch.no_grad():
            mask    = self.U.abs() > 1.0
            U_thres = self.U.clone()
            U_thres[mask] = 1 + 0j

            if mode == "far":
                return U_thres
            
            U_masked = self.U * (~mask).float()
            
            E_masked = torch.fft.fftshift(
                torch.fft.ifft2(torch.fft.ifftshift(U_masked), norm="ortho")
            )

            if mode == "slm":
                return E_masked
            if mode == "both":
                return E_masked, U_thres
            else:
                raise ValueError(f"Invalid mode '{mode}'; must be 'slm', 'far', or 'both'.")

    # --- PhysicalModule interface ---

    def rebuild(self):
        """
        Rebuild the module after changing the geometry or the shape of the learnable field.
        Both the far-field sampling (M, N) and the far-field window (fov) can change, 
        and they require different treatments.

        NOTE: interpolating stray light across a sampling change is approximate
        solution. If the background matters quantitatively, refit it
        at the intended sampling.
        """
        def _centre_resize(x: torch.Tensor, target_shape: tuple) -> torch.Tensor:
            """
            Centre zero-pad and/or centre-crop the last two dims to target_shape.
            """
            *batch, H, W = x.shape
            Ht, Wt = target_shape
            if (H, W) == (Ht, Wt):
                return x
            out = x.new_zeros(*batch, Ht, Wt)
            h, w = min(H, Ht), min(W, Wt)
            src_t, src_l = (H - h) // 2, (W - w) // 2
            dst_t, dst_l = (Ht - h) // 2, (Wt - w) // 2
            out[..., dst_t:dst_t + h, dst_l:dst_l + w] = x[..., src_t:src_t + h, src_l:src_l + w]
            return out


        def _resample_far_field(U: torch.Tensor, target_shape: tuple) -> torch.Tensor:
            """
            Resample a centred far-field array onto a different number of samples
            covering the SAME angular window (i.e. a change of M, N at fixed fov).
            Uses band-limited sinc-interpolation.
            """
            if tuple(U.shape[-2:]) == tuple(target_shape):
                return U
            c = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(U)))
            c = _centre_resize(c, target_shape)
            return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(c)))

        g = self.geometry
        if self._full_res:
            old_MN = getattr(self, "_field_MN", (g.M, g.N))
            # Where the current samples land on the new sampling grid: the
            # window is unchanged, so the sample count scales with M, N.
            resampled_shape = (
                max(1, int(round(self.U.shape[0] * g.M / old_MN[0]))),
                max(1, int(round(self.U.shape[1] * g.N / old_MN[1]))),
            )
            if tuple(self.U.shape) != tuple(resampled_shape) or self.U.shape != g.far_fov_samples:
                with torch.no_grad():
                    U_new = self.U.detach()
                    if tuple(U_new.shape) != tuple(resampled_shape):
                        if U_new.abs().max() > 1.0:
                            warnings.warn(
                                "Resampling a background field that contains saturation "
                                "markers (|U| > 1); band-limited interpolation will ring "
                                "around them. Refit the background at the new far-field "
                                "sampling if it matters."
                            )
                        U_new = _resample_far_field(U_new, resampled_shape)
                    # then the fov change: crop/zero-pad about the centre
                    U_new = _centre_resize(U_new, g.far_fov_samples)
                    self.U = nn.Parameter(U_new.contiguous())
            self._shape = g.far_fov_samples
        self._field_MN = (g.M, g.N)

        super().rebuild()
        
    def _forward_disabled(self) -> torch.Tensor:
        """
        No stray light: return a zero field of the correct far-field shape.
        This is added to U in HoloSystem.propagate(), so zero is the correct identity.
        """
        return torch.zeros(
            self.geometry.far_fov_samples,
            dtype=torch.cfloat,
            device=self.geometry.device,
        )

    def _forward_enabled(self) -> torch.Tensor:
        return self.field  # returns the learnable far-field background E


# =============================================================================
# CameraModule - affine warp between far-field and camera plane, and saturation
# =============================================================================

class CameraModule(PhysicalModule):
    """
    Observation model: Simulates the observation of a far-field diffraction pattern 
    by a physical camera sensor using a decoupled, differentiable affine warp.
 
    The affine transformation maps Camera Normalized Coordinates [-1, 1] to 
    Far-Field Normalized Coordinates [-1, 1] to satisfy PyTorch's backward-lookup 
    convention (F.affine_grid and F.grid_sample).

    To guarantee a highly convex loss landscape and prevent cross-talk during optimization, 
    the parameters are mathematically decoupled: changing scale or rotation adjusts the 
    framing strictly *around* the camera center, without shifting its position.

    Physical Parameter Definitions:
    ------------------------------
    - rotation (float): 
        Rotation angle in radians around the camera's optical axis. 
        A positive rotation physically rotates the camera counter-clockwise, 
        causing the observed far-field scene to appear rotated clockwise.
    - scale (tuple of float): 
        Zoom multipliers [sx, sy] relative to the ideal physical setup.
        A positive scale (> 1.0) increases the camera's field of view (zooms out), 
        making objects in the far-field appear smaller on the sensor.
    - shift (tuple of float): 
        Translation [dx, dy] in raw camera pixel units. 
        A positive shift physically offsets the camera body down and to the right, 
        causing the observed far-field pattern to appear shifted up and to the left.
    """
 
    def __init__(self, geometry: OpticsGeometry):
        super().__init__(geometry)   # calls _build_static_buffers
 
        g = geometry
        # --- learnable raw parameters (unconstrained) ---
        self.rotation_raw = nn.Parameter(torch.tensor(0.0, device=g.device))
        self.scale_raw    = nn.Parameter(torch.zeros(2,    device=g.device))
        self.shift_raw    = nn.Parameter(torch.zeros(2,    device=g.device))

    # --- property interface ---
    @property
    def matrix(self) -> torch.Tensor:
        """3x3 affine matrix: camera normalised ← far-field normalised."""
        if self.is_frozen:
            return self._matrix_buffer
        else:
            return self._compute_matrix()
        
    @property
    def affine_params(self) -> dict:
        """Current affine parameters in physical units (radians, scale, pixels)."""
        rot_rad, scale, shift = self._physical_params()
        return {
            "rotation": rot_rad.item(),
            "scale":    scale.tolist(),
            "shift":    shift.tolist(),
        }
 
    @affine_params.setter
    def affine_params(self, p: dict):
        """Set parameters from physical units; updates the cached matrix."""
        g = self.geometry
        with torch.no_grad():
            rot_rad = p.get("rotation", 0.0)
            rot = rot_rad / g.affine_rotation_range
            if abs(rot) >= 0.999:
                warnings.warn(
                    f"Camera rotation {rot_rad} exceeds range {g.affine_rotation_range}; clipping."
                )
                rot = np.clip(rot, -0.999, 0.999)
                
            self.rotation_raw.copy_(
                torch.atanh(torch.tensor(rot, device=g.device))
            )
            scale = p.get("scale", [1.0, 1.0])
            sx = (scale[0] - 1.0) / g.affine_scale_range
            sy = (scale[1] - 1.0) / g.affine_scale_range
            if np.any(np.abs([sx, sy]) >= 0.999):
                warnings.warn(
                    f"Camera scale {scale} exceeds range {g.affine_scale_range}; clipping."
                )
                sx = np.clip(sx, -0.999, 0.999)
                sy = np.clip(sy, -0.999, 0.999)

            self.scale_raw.copy_(
                torch.atanh(torch.tensor([sx, sy], device=g.device))
            )
            shift = p.get("shift", [0.0, 0.0])
            shx = shift[0] / g.affine_shift_range
            shy = shift[1] / g.affine_shift_range
            if np.any(np.abs([shx, shy]) >= 0.999):
                warnings.warn(
                    f"Camera shift {shift} exceeds range {g.affine_shift_range}; clipping."
                )
                shx = np.clip(shx, -0.999, 0.999)
                shy = np.clip(shy, -0.999, 0.999)
            
            self.shift_raw.copy_(
                torch.atanh(torch.tensor([shx, shy], device=g.device))
            )
        
        if self.is_frozen:
            self._clear_cached_buffers()
            self._build_cached_buffers()
    
    # ------------------------------------------------------------------
    # Coordinate conversions (align_corners=False convention, matches
    # F.affine_grid / F.grid_sample used in _affine())
    # ------------------------------------------------------------------

    @staticmethod
    def _pix_to_norm(coord, size: int):
        return (2.0 * coord + 1.0) / size - 1.0

    def far_pixel_to_norm(self, points_xy: torch.Tensor) -> torch.Tensor:
        """(N, 2) far-field pixel (x, y) -> (N, 2) far-field normalised [-1, 1]."""
        Mf, Nf = self.geometry.far_fov_samples
        x = self._pix_to_norm(points_xy[..., 0], Nf)
        y = self._pix_to_norm(points_xy[..., 1], Mf)
        return torch.stack([x, y], dim=-1)

    def camera_pixel_to_norm(self, points_xy: torch.Tensor) -> torch.Tensor:
        """(N, 2) camera pixel (x, y) -> (N, 2) camera normalised [-1, 1]."""
        Hc, Wc = self.geometry.camera_shape
        x = self._pix_to_norm(points_xy[..., 0], Wc)
        y = self._pix_to_norm(points_xy[..., 1], Hc)
        return torch.stack([x, y], dim=-1)

    # ------------------------------------------------------------------
    # User interface 
    # ------------------------------------------------------------------

    def set_affine_from_correspondence(self, matrix_2x3) -> dict:
        """
        Set rotation/scale/shift directly from a fitted 2x3 matrix A with
            far_norm ≈ A @ [cam_norm, 1]
        
        Decomposes the matrix using the updated decoupled transformation convention:
            row0 = [ s_x * cos_r, -s_y * sin_r,  shift_norm_x]
            row1 = [ s_x * sin_r,  s_y * cos_r,  shift_norm_y]
        where:
            shift_norm_x = shift[0] * (2.0 / Wc) * base_sx

        Parameters
        ----------
        matrix_2x3 : array-like, shape (2, 3)
            Fitted affine matrix mapping camera normalized coordinates to far-field normalized coordinates.
        """
        A = np.asarray(matrix_2x3, dtype=np.float64)
        a, b, tx = A[0]
        c, d, ty = A[1]

        base_sx, base_sy = self._get_base_scale()

        # 1. Recover scale factors (s_x, s_y) from columns
        s_x = np.hypot(a, c)
        s_y = np.hypot(b, d)
        
        # 2. Recover rotation angle matching the new [cos, -sin; sin, cos] matrix setup
        r1 = np.arctan2(c, a)
        r2 = np.arctan2(-b, d)
        rot = np.arctan2(np.sin(r1) + np.sin(r2), np.cos(r1) + np.cos(r2))

        # 3. De-process the shift parameter
        # Since shift_norm_x = shift[0] * (2.0 / Wc) * base_sx, 
        # we isolate shift by dividing out the base_scale constants.
        Hc, Wc = self.geometry.camera_shape
        
        shift_x = (tx / base_sx) * (Wc / 2.0)
        shift_y = (ty / base_sy) * (Hc / 2.0)

        params = {
            "rotation": float(rot),  # radians
            "scale":    [float(s_x / base_sx), float(s_y / base_sy)],
            "shift":    [float(shift_x), float(shift_y)],
        }
        
        self.affine_params = params
        return params

    def shift_in_far_field(self, u: float, v: float):
        """
        Physically offset the camera position by an explicit coordinate displacement 
        (u, v) given in the reference frame of the Far-Field.

        Parameters
        ----------
        u, v (float): 
            Displacements normalized to the physical boundaries of the first 
            diffraction order, where [-0.5, 0.5] represents the span of the 
            ideal first-order window.
        
        Operation
        ----------
        Inverts the module's static geometric mapping to translate the far-field 
        coordinates back into raw camera pixel units, updating the underlying 
        differentiable parameters cleanly without corrupting learnable scale or 
        rotation states.
        """
        g = self.geometry
        Hc, Wc = g.camera_shape

        # 1. Get the static base scales (these map camera space to the full FOV)
        base_sx, base_sy = self._get_base_scale()
        
        # 2. Convert u, v (which are in [-0.5, 0.5] of the first order) 
        # into the full normalized far-field coordinate space [-1, 1] 
        # by dividing out the g.fov multiplier.
        u_ff_norm = (u / 0.5) / g.fov
        v_ff_norm = (v / 0.5) / g.fov

        # 3. Invert the matrix math to calculate how many camera pixels this represents
        # shift_norm = shift_px * (2.0 / Wc) * base_s  ==>  shift_px = shift_norm * (Wc / 2.0) / base_s
        delta_shift_px_x = (u_ff_norm * (Wc / 2.0)) / base_sx
        delta_shift_px_y = (v_ff_norm * (Hc / 2.0)) / base_sy

        # 4. Update the parameters 
        affine_params = self.affine_params
        new_shift = [
            affine_params["shift"][0] - delta_shift_px_x,
            affine_params["shift"][1] - delta_shift_px_y,
        ]

        self.affine_params = {
            "rotation": affine_params["rotation"],
            "scale": affine_params["scale"],
            "shift": new_shift,
        }
 
    # ------------------------------------------------------------------
    # Buffer management for frozen parameters
    # ------------------------------------------------------------------
 
    def _build_cached_buffers(self):
        self.register_buffer("_matrix_buffer", self._compute_matrix().detach())

    def _clear_cached_buffers(self):
        if "_matrix_buffer" in self._buffers:
            del self._buffers["_matrix_buffer"]

    # ------------------------------------------------------------------
    # Parameter → physical units
    # ------------------------------------------------------------------
 
    def _physical_params(self):
        """
        Convert unconstrained raw scalars into physically meaningful units.
        Rotation is in radians, scale is a multiplier, and shift is in camera pixels.
        """
        g     = self.geometry

        scale = 1.0 + g.affine_scale_range * torch.tanh(self.scale_raw)
        rot_rad   = g.affine_rotation_range * torch.tanh(self.rotation_raw)
        shift = g.affine_shift_range    * torch.tanh(self.shift_raw)

        return rot_rad, scale, shift
    
    def _get_base_scale(self):
        """
        Compute the ideal, static scaling factor mapping the camera sensor 
        dimensions to the full physical Far-Field Field of View (FoV).

        This represents the baseline mapping of a perfectly aligned system 
        with zero physical perturbations (no scale deviation, rotation, or shift).

        Coordinate Conventions
        -----------------------
        - Camera Normalized Space: [-1, 1] maps to the physical edges of the camera sensor.
        - Far-Field Normalized Space: [-1, 1] maps to the full numerical FoV (First Order * g.fov).
        
        Returns
        --------
        base_sx, base_sy (float, float): 
            The static horizontal and vertical scale ratios.
        """
        g = self.geometry
        Hc, Wc = g.camera_shape
        
        # FoV of the first diffraction order in physical units
        fov_first_phys = g.focal_length * g.wavelength / g.slm_pixel_pitch
        
        # Full numerical FoV in physical units
        fov_phys = fov_first_phys * g.fov

        cam_phys_w = Wc * g.camera_pixel_pitch
        cam_phys_h = Hc * g.camera_pixel_pitch
        
        base_sx = cam_phys_w / fov_phys
        base_sy = cam_phys_h / fov_phys
        
        return base_sx, base_sy
    
    def _compute_matrix(self) -> torch.Tensor:
        """
        Builds the 3x3 homogeneous affine transformation matrix mapping 
        Camera Normalized Space [-1, 1] -> Far-Field Normalized Space [-1, 1].

        Optimization Architecture (Decoupled Chain)
        -------------------------------------------
        Implements the matrix chain: M = T_static * S_learnable * R_learnable
        
        By applying translation on the leftmost side using only static geometric 
        constants, the learnable scale and rotation parameters operate strictly 
        around the camera's local origin. This eliminates gradient cross-talk, 
        preventing changes in zoom/rotation from causing unintended scene drifting 
        during training.

        Returns
        --------
        matrix (torch.Tensor): 
            A 3x3 tensor representing the destination-to-source affine warp.
        """
        g = self.geometry
        Hc, Wc = g.camera_shape
        rot, scale, shift = self._physical_params()

        # 1. Determine the CONSTANT ideal scale factor from the physical geometry
        base_sx, base_sy = self._get_base_scale()

        # 2. Convert Shift from Camera Pixels to Far-Field space using ONLY the base constants.
        # This keeps 'shift' in camera pixels, but makes it completely independent 
        # of the learnable scale and rotation parameters during optimization.
        shift_norm_x = shift[0] * (2.0 / Wc) * base_sx
        shift_norm_y = shift[1] * (2.0 / Hc) * base_sy

        # 3. Combine Base Scale and Learnable Perturbation Scale
        s_x = base_sx * scale[0]
        s_y = base_sy * scale[1]
        
        # 4. Construct the standalone matrix
        cos_r, sin_r = torch.cos(rot), torch.sin(rot)
        
        # By leaving shift_norm standalone here, rotation and scale occur 
        # perfectly around the camera's center coordinate.
        row0 = torch.stack([ s_x * cos_r, -s_y * sin_r, shift_norm_x])
        row1 = torch.stack([ s_x * sin_r,  s_y * cos_r, shift_norm_y])
        row2 = torch.tensor([0.0, 0.0, 1.0], device=g.device)
        
        return torch.stack([row0, row1, row2])
 
    # ------------------------------------------------------------------
    # PhysicalModule interface
    # ------------------------------------------------------------------
 
    def _forward_disabled(self, I: torch.Tensor) -> torch.Tensor:
        """No observation model: return far-field intensity unchanged."""
        return I

    def _pixel_integration(self, I: torch.Tensor, 
                           matrix: torch.Tensor, 
                           camera_shape: tuple, 
                           ss: int = 2
                           ) -> torch.Tensor:
        """
        Integrate the far-field intensity I over camera pixels using supersampling.

        Parameters
        ----------
        I : (B, Mf, Nf) float — far-field intensity
        matrix : (2, 3) float — affine transformation matrix mapping camera normalized coordinates to far-field normalized coordinates
        camera_shape : tuple of ints — (Hc, Wc) camera image shape
        ss : int — supersampling factor

        Returns
        -------
        C : (B, Hc, Wc) float — camera image
        """
        Hc, Wc = camera_shape

        I_hi = (F.interpolate(I.unsqueeze(0), scale_factor=ss,
                               mode="bilinear", align_corners=False)
                if ss > 1 else I.unsqueeze(0))
        grid  = F.affine_grid(matrix.unsqueeze(0),
                               size=(1, 1, Hc * ss, Wc * ss),
                               align_corners=False)
        C   = F.grid_sample(I_hi, grid, mode="bilinear",
                               padding_mode="zeros", align_corners=False)
        if ss > 1:
            C = F.avg_pool2d(C, kernel_size=ss, stride=ss)
        return C.squeeze(0)
 
    def _affine(self, I: torch.Tensor, camera_shape: Optional[tuple] = None) -> torch.Tensor:
        """
        Apply the affine warp to the far-field intensity I and integrate over camera pixels.

        Parameters
        ----------
        I : (B, Mf, Nf) float — far-field intensity
        camera_shape : tuple of ints, optional — (Hc, Wc) camera image shape; 
            if None, uses the geometry's camera shape
            if provided, the affine matrix is adjusted to account for the difference in resolution.

        Returns
        -------
        C : (B, Hc, Wc) float — camera image
        """
        ss = self.geometry.camera_affine_supersample

        if camera_shape is not None:
            Htrain, Wtrain = self.geometry.camera_shape
            Hfull, Wfull = camera_shape

            sx = Wtrain / Wfull
            sy = Htrain / Hfull

            S = torch.tensor([
                [1/sx, 0, 0],
                [0, 1/sy, 0],
                [0, 0, 1],
            ], device=self.geometry.device)

            matrix = (self.matrix @ S)[:2]
            c_shape = camera_shape
        else:
            # Get the affine matrix based on rotation_raw / scale_raw / shift_raw. 
            matrix = self.matrix[:2] 
            c_shape = self.geometry.camera_shape   

        C = self._pixel_integration(I, matrix, c_shape, ss=ss)
        return C

    def _saturate(self, C: torch.Tensor) -> torch.Tensor:
        """
        Apply saturation clipping to the camera image.

        Parameters
        ----------
        C : (B, Hc, Wc) float — camera image

        Returns
        -------
        C_sat : (B, Hc, Wc) float — saturated camera image
        """
        C_hard = torch.clamp(C, max=1.0)
        
        # straight-through estimator: hard clip in forward, soft in backward
        return C + (C_hard - C).detach()

    def _forward_enabled(self, I: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        I : (B, Mf, Nf) float — far-field intensity
 
        Returns
        -------
        S : (B, Hc, Wc) float — warped camera image with saturation clipping
        """
        C = self._affine(I)
        S = self._saturate(C)
        return S
 
    # ------------------------------------------------------------------
    # Diagnostics and User Interface
    # ------------------------------------------------------------------

    def affine_inverse(self, C: torch.Tensor, square: bool = True) -> torch.Tensor:
        """
        Apply affine transfrom from the camera space to the far-field space. 
        We don't integrate, it just returns the warped image in the far-field space.

        Parameters
        ----------
        C : (B, Hc, Wc) float - camera image. Can be different size than the camera shape used for training.

        Returns
        -------
        I : (B, M_max, M_max) float - warped far-field image sampled in the numerical far-field space within the FoV.
            where M_max is the maximum of the far-field samples (Mf, Nf) defined in the geometry, to give a square output.
        """
        g = self.geometry

        Htrain, Wtrain = g.camera_shape
        Hfull, Wfull = C.shape[-2:]

        sx = Wtrain / Wfull
        sy = Htrain / Hfull

        S = torch.tensor([
            [1/sx, 0, 0],
            [0, 1/sy, 0],
            [0, 0, 1],
        ], device=self.geometry.device)

        matrix = (self.matrix @ S).inverse()[:2]

        # sample this at the square samples, instead of the original fov sampling
        if square:
            M = max(*self.geometry.far_fov_samples)
            N = M
        else:
            M, N = self.geometry.far_fov_samples
        I = self._pixel_integration(C, matrix, (M, N), ss=1)

        return I
        
    def get_corners(self):
        """
        Return the camera frame corners mapped back into normalised far-field
        space, alongside the far-field frame corners.  Used for plotting.
        """
        g      = self.geometry
        matrix = self.matrix
        fov = g.fov

        # Perform the computation in the normalised camera space, then map to far-field space using the affine matrix.
        corners_cam_cam = torch.tensor(
            [[-1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1], [-1, -1, 1]],
            dtype=torch.float, device=g.device,
        )

        corners_cam_far = (matrix @ corners_cam_cam.T).T[:, :2]
        
        # compute the corner coordinates in the far-field plane 
        # (normalised to [-0.5, 0.5] at the edges of the first diffraction order)
        corners_fov_far = torch.tensor(
            [[-1, -1], [-1, 1], [1, 1], [1, -1], [-1, -1]],
            dtype=torch.float, device=g.device,
        )

        corners_cam = corners_cam_far * fov * 0.5
        corners_fov = corners_fov_far * fov * 0.5

        return corners_cam , corners_fov


# =============================================================================
# HoloSystem — top-level digital twin
# =============================================================================

@dataclass
class ModuleFlags:
    """Flags to control the modules in HoloSystem.
    
    Setting a flag to True or False overrides the default behavior.
    Leaving it as None will defer to the method's default handler.
    """
    lut: Optional[bool] = None
    pixel: Optional[bool] = None
    slm_field: Optional[bool] = None
    pupil: Optional[bool] = None
    background: Optional[bool] = None
    camera: Optional[bool] = None


class HoloSystem(nn.Module):
    """
    Full holographic digital twin: SLM → far-field propagation → camera.

    All physical modules are always instantiated.  Use .enable() / .disable()
    on each child module to control its contribution, and .freeze() / .unfreeze()
    to control whether its parameters are optimised.  The forward() method
    contains no flag checks. Use .save(), .load(), and .copy() to checkpoint the entire system.
    Use .report() to get a figure showing of the current state of the system.

    Parameters
    ----------
    geometry       : OpticsGeometry
    lut_n_bins     : number of LUT bins
    lut_use_depth  : whether to add a per-pixel depth map to the LUT
    lut_init_scale : initial LUT phase span in radians
    lut_depth_grid_divisor : LUT depth grid divisor
    lut_init_depth_map : initial LUT depth map (y, x). Gets interpolated to the LUT grid
    crosstalk_init_sigma    : initial Gaussian σ (y, x) in pixels
    crosstalk_pixel_samples : sub-pixel oversampling K
    crosstalk_kernel_pixels : kernel half-width in pixels
    crosstalk_use_residual  : add learnable residual kernel
    crosstalk_fast_approx   : use FFT-crop rather than full CZT
    pixel_use_deadspace     : whether to model dead-space between pixels
    pixel_deadspace_reflectance : reflectance of dead-space (0.0-1.0)
    pupil_num_tiles         : OTF tiles per far-field dimension
    pupil_init_coeffs       : SeidelCoefficients initial values
    pupil_use_vignetting    : whether to apply vignetting mask
    pupil_init_aperture_distance : initial distance from SLM to aperture plane
    pupil_init_aperture_offset   : initial (y, x) offset of the aperture
    slm_field_init_field    : complex Tensor for the incident field
    slm_field_use_scale     : whether to apply the energy scale
    background_init_field   : complex Tensor for the initial background
    pupil_cache_otf         : whether to cache the OTF for speed
    chunk_bytes             : memory tuning for FFT chunking
    verbose                 : print diagnostic messages
    """

    def __init__(
        self,
        geometry: OpticsGeometry,
        *,
        # LUT
        lut_n_bins:          int   = 10,
        lut_use_depth:       bool  = False,
        lut_init_scale:      float = 2 * np.pi,
        lut_depth_grid_divisor: int = 40,
        lut_init_depth_map: Optional[torch.Tensor] = None,
        # Crosstalk
        crosstalk_pixel_samples: int   = 3,
        crosstalk_kernel_pixels: int   = 3,
        crosstalk_use_residual:  bool  = True,
        crosstalk_fast_approx:   bool  = True,
        crosstalk_init_sigma:    tuple = (0.3, 0.3),
        # Envelope
        pixel_init_deadspace_reflectance: float = 0.8,
        # Pupil
        pupil_num_tiles:     int   = 40,
        pupil_init_coeffs:       Optional[AberrCoefficients] = None,
        pupil_use_vignetting:    bool = False,
        pupil_init_aperture_distance: Optional[float] = None,
        pupil_init_aperture_offset:   Optional[tuple] = (0.0, 0.0),
        # SLM field
        slm_field_init_field:  Optional[torch.Tensor] = None,
        slm_field_use_scale:   bool = False,
        # Background
        background_init_field: Optional[torch.Tensor] = None,
        # Memory tuning (see PixelModule.fft_chunk_bytes / PupilModule.tile_chunk_bytes)
        chunk_bytes: int = 128 << 20,
        pupil_cache_otf: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        self.geometry: OpticsGeometry = geometry
        self._verbose  = verbose

        # Store the initialisation kwargs for later use in .save_checkpoint()
        local_vars = locals()
        self._set_init_kwargs(local_vars)

        self.lut = LUTModule(
            geometry,
            n_bins=lut_n_bins,
            use_depth=lut_use_depth,
            init_scale=lut_init_scale,
            depth_grid_divisor=lut_depth_grid_divisor,
            init_depth_map=lut_init_depth_map,
        )

        self.pixel = PixelModule(
            geometry,
            init_sigma=crosstalk_init_sigma,
            init_deadspace_reflectance=pixel_init_deadspace_reflectance,
            pixel_samples=crosstalk_pixel_samples,
            kernel_pixels=crosstalk_kernel_pixels,
            use_residual=crosstalk_use_residual,
            fast_approx=crosstalk_fast_approx,
            pupil_num_tiles=pupil_num_tiles,
            fft_chunk_bytes=chunk_bytes,
        )

        self.slm_field = SLMFieldModule(
            geometry,
            init_field=slm_field_init_field,
            use_scale=slm_field_use_scale,
        )

        self.pupil = PupilModule(
            geometry,
            init_coeffs=pupil_init_coeffs,
            num_tiles=pupil_num_tiles,
            use_vignetting=pupil_use_vignetting,
            init_aperture_distance=pupil_init_aperture_distance,
            init_aperture_offset=pupil_init_aperture_offset,
            tile_chunk_bytes=chunk_bytes,
            cache_otf=pupil_cache_otf,
        )

        self.background = BackgroundModule(geometry, init_field=background_init_field)
        self.camera     = CameraModule(geometry)

        # registry of all modules for convenience
        self._managed_modules = ['lut', 'pixel', 'slm_field', 'pupil', 'background', 'camera']

    def _set_init_kwargs(self, kwargs):
        """
        Store the initialisation kwargs for later use in .save_checkpoint().
        Store only the keys that define the parameterisation of the optics modules, 
        but not the initial fields themselves, as those are not needed for re-initialisation.
        """
        _CONFIG_KEYS = (
            "lut_n_bins", "lut_use_depth",
            "lut_depth_grid_divisor", "lut_init_depth_map",
            "crosstalk_pixel_samples", "crosstalk_kernel_pixels",
            "crosstalk_use_residual", "crosstalk_fast_approx",
            "pupil_num_tiles", "pupil_use_vignetting",
            "chunk_bytes", "pupil_cache_otf",
            "slm_field_use_scale", "slm_field_remove_ramp", 
            "verbose",
        )
        self._init_kwargs = {k: kwargs[k] for k in _CONFIG_KEYS if k in kwargs}

    # ------------------------------------------------------------------
    # Convenience: operate on all optics modules at once
    # ------------------------------------------------------------------

    def _compute_target_states(self, flags: Optional[ModuleFlags] = None, default: Optional[bool] = None) -> dict:
        """The single source of truth for parameter parsing using ModuleFlags."""
        # Initialize everything with the base default value
        states = {k: default for k in self._managed_modules}
        
        # Override defaults with explicitly provided dataclass flags
        if flags is not None:
            for field in fields(flags):
                val = getattr(flags, field.name)
                if val is not None:
                    states[field.name] = val
                    
        return states

    def modules(self, default: Optional[bool] = True, flags: Optional[ModuleFlags] = None) -> list:
        """
        Return a list of modules filtered by the provided flags and default behavior.

        Parameters
        ----------
        default : Optional[bool]
            If True, return all modules.
            If False, return no modules.
            If None, return only modules explicitly named in flags.
        flags : Optional[ModuleFlags]
            If provided, overrides the default behavior for specific modules.
        """
        states = self._compute_target_states(flags=flags, default=default)
        
        if default is not None:
            return [getattr(self, k) for k, v in states.items() if v]
        
        # If default is None, only return modules explicitly named (not None) in flags
        return [
            getattr(self, k) for k, v in states.items() 
            if flags is not None and getattr(flags, k) is not None
        ]

    def params(self, default: Optional[bool] = True, flags: Optional[ModuleFlags] = None) -> list:
        """
        Return a list of all parameters for the filtered modules.

        Parameters
        ----------
        default : Optional[bool]
            If True, return parameters for all enabled modules.
            If False, return parameters for all disabled modules.
            If None, return parameters for all modules explicitly named in flags.
        flags : Optional[ModuleFlags]
            If provided, overrides the default behavior for specific modules.
        """
        return [p for m in self.modules(flags=flags, default=default) for p in m.parameters()]

    def set_active(self, default: Optional[bool] = None, flags: Optional[ModuleFlags] = None) -> "HoloSystem":
        """
        Unified control for enabling/disabling modules.

        Parameters
        ----------
        default : Optional[bool]
            If True, all modules will be enabled.
            If False, all modules will be disabled.
            If None, the default behavior is to leave modules unchanged.
        flags : Optional[ModuleFlags]
            If provided, overrides the default behavior for specific modules.
        """
        states = self._compute_target_states(flags=flags, default=default)
        for name, should_enable in states.items():
            if should_enable is not None:
                getattr(self, name).enable() if should_enable else getattr(self, name).disable()
        return self

    def set_unfrozen(self, default: Optional[bool] = None, flags: Optional[ModuleFlags] = None) -> "HoloSystem":
        """
        Unified control for freezing/unfreezing modules.
        
        Parameters
        ----------
        default : Optional[bool]
            If True, all modules will be unfrozen (trainable). 
            If False, all modules will be frozen (non-trainable). 
            If None, the default behavior is to leave modules unchanged
        flags : Optional[ModuleFlags]
            If provided, overrides the default behavior for specific modules.
        """
        states = self._compute_target_states(flags=flags, default=default)
        for name, should_unfreeze in states.items():
            if should_unfreeze is not None:
                getattr(self, name).unfreeze() if should_unfreeze else getattr(self, name).freeze()
        return self

    # ------------------------------------------------------------------
    # Forward — no flag inspection, no if/else
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_batch(g: torch.Tensor) -> torch.Tensor:
        return g.unsqueeze(0) if g.ndim == 2 else g

    def propagate(self, g: torch.Tensor) -> torch.Tensor:
        """
        SLM grayscale pattern → far-field complex amplitude.

        Parameters
        ----------
        g : (B, P, Q) or (P, Q) float in [0, 1]

        Returns
        -------
        U : (B, Mf, Nf) complex
        """
        g   = self._ensure_batch(g).to(self.geometry.device)
        phi = self.lut(g)           # (B, P, Q) → (B, P, Q)   phase
        E_in = self.slm_field()        # (B, P, Q)   incident field
        U_ideal = self.pixel(phi, E_in)   # (B, P, Q) → (B, Mf, Nf) ideal far-field
        U_pupil = self.pupil(U_ideal) # (B, Mf, Nf) → (B, Mf, Nf) pupil-modified far-field
        U = U_pupil + self.background()   # zero when background is disabled
        return U
    
    def get_intensity(self, U: torch.Tensor) -> torch.Tensor:
        """
        Far-field complex amplitude -> far-field intensity.

        Parameters
        ----------
        U : (B, Mf, Nf) complex

        Returns
        -------
        I : (B, Mf, Nf) float — far-field intensity in [0, 1]
        """
        return U.real ** 2 + U.imag ** 2

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        """
        SLM grayscale pattern → predicted camera image.

        Parameters
        ----------
        g : (B, H, W) or (H, W) float in [0, 1]

        Returns
        -------
        S : (B, Hc, Wc) float — camera intensity in [0, 1]
        """
        U = self.propagate(g)
        I = self.get_intensity(U)
        return self.camera(I)  # identity when camera is disabled
    
    # ------------------------------------------------------------------
    # User utility methods
    # ------------------------------------------------------------------
    def set_fov(self, fov: float):
        """
        Set the field of view (FOV) in multiples of the first order FoV. Updates the geometry and all
        dependent modules.

        Parameters
        ----------
        fov : float — new FOV in multiples of the first order FoV
        """
        # geometry is immutable, so we need to copy the geometry and only change the fov in the input
        new_geometry = dataclasses.replace(self.geometry, fov=fov)
        self.geometry = new_geometry
        modules = self.modules()
        for m in modules:
            m.geometry = new_geometry
            m.rebuild()

    def square_far_field(self, square: bool = True):
        """
        Set whether the far-field output should be square or rectangular. Updates the geometry and all
        dependent modules.

        Parameters
        ----------
        square : bool — whether to make the far-field output square
        """
        new_geometry = dataclasses.replace(self.geometry, square_far_field=square)
        self.geometry = new_geometry
        modules = self.modules()
        for m in modules:
            m.geometry = new_geometry
            m.rebuild()

    def remove_phase_gauge(self):
        """
        Remove the global linear phase from the SLM field and apply the corresponding
        affine shift to the camera module. This is used to remove the gauge invariance
        between the SLM field phase ramp and the camera affine shift, which can help
        the training converge faster.
        """
        g = self.geometry

        # Remove the global linear phase ramp from the SLM field and get the 
        # corresponding slopes in radians per pixel (sx, sy)
        # It is in the range of [-pi, pi] within the first order
        sx, sy = self.slm_field.remove_ramp()

        # Convert to normalised frequency coordinates (u0, v0) in [-0.5, 0.5] 
        # where 1 corresponds to the edges of the first diffraction order
        u0 = sx.item() / (2 * torch.pi)
        v0 = sy.item() / (2 * torch.pi)

        # Apply the corresponding affine shift to the camera module
        self.camera.shift_in_far_field(u0, v0)

    # ------------------------------------------------------------------
    # Checkpoint save / load / copy
    # ------------------------------------------------------------------
    
    def checkpoint(self) -> dict:
        """
        Return a self-contained checkpoint dict of the current geometry and active modules.

        Contents
        --------
        state_dict : standard PyTorch state dict (Parameters only, buffers are derived)
        active     : dict of {module_name: bool} enable/disable flags
        frozen     : dict of {module_name: bool} freeze/unfreeze flags
        geometry   : dataclasses.asdict(self.geometry) — all grid/physical constants
        config     : dict of initialization kwargs for architecture reproduction
        """
        return {
            "state_dict": {
                name: p.detach().clone()
                for name, p in self.named_parameters()
            },
            "active": {
                "lut": self.lut.is_active, "pixel": self.pixel.is_active,
                "slm_field": self.slm_field.is_active, "pupil": self.pupil.is_active,
                "background": self.background.is_active, "camera": self.camera.is_active,
            },
            "frozen": {
                "lut": self.lut.is_frozen, "pixel": self.pixel.is_frozen,
                "slm_field": self.slm_field.is_frozen, "pupil": self.pupil.is_frozen,
                "background": self.background.is_frozen, "camera": self.camera.is_frozen,
            },
            "geometry": dataclasses.asdict(self.geometry),
            "config": self._init_kwargs,
        }

    def save(self, path: str):
        """
        Save a self-contained, weights_only-safe checkpoint to *path*.

        Contains only plain-data types (tensors, bools, dict) — no pickled
        class objects — so it loads safely under torch.load(weights_only=True).
        """
        torch.save(self.checkpoint(), path)

    @classmethod
    def load(cls, path: str, device: str = None, **override_kwargs) -> "HoloSystem":
        """
        Reconstruct a HoloSystem from a checkpoint saved by .save().

        Parameters
        ----------
        path     : path to the .pt file produced by save()
        device   : override the device stored in the checkpoint geometry
                (enables loading a GPU-trained model on CPU, etc.)
        **override_kwargs : any HoloSystem __init__ kwarg to override
                            (e.g. pupil_num_tiles=40)
        """
        payload = torch.load(path, map_location=device, weights_only=True)

        geom_dict = {
            k: v for k, v in payload["geometry"].items()
            if k in OpticsGeometry.__dataclass_fields__
            and k not in ("P", "Q", "M", "N", "Pf", "Qf", "Mf", "Nf")
        }
        if device is not None:
            geom_dict["device"] = device
        geometry = OpticsGeometry(**geom_dict)

        config = dict(payload.get("config", {}))
        config.update(override_kwargs)

        # Drop config keys that no longer exist as constructor arguments, so
        # that checkpoints written by an earlier version still load.
        import inspect
        valid = set(inspect.signature(cls.__init__).parameters)
        retired = sorted(k for k in config if k not in valid)
        if retired:
            warnings.warn(
                f"Ignoring retired checkpoint config key(s): {', '.join(retired)}."
            )
            config = {k: v for k, v in config.items() if k in valid}

        system = cls(geometry, **config)
        # strict=False: buffers present in `system` but absent from the
        # checkpoint are left as freshly (re)built — exactly what we want.
        # explicitly set the SLM field shape to match the checkpoint, since it is a buffer and not a parameter.
        system.slm_field.set_shape((payload["state_dict"]["slm_field.E"].shape))
        system.load_state_dict(payload["state_dict"], strict=False)

        for name, active in payload.get("active", {}).items():
            m = getattr(system, name, None)
            if m is not None:
                m.enable() if active else m.disable()

        # Set the frozen status of each module
        for name, frozen in payload.get("frozen", {}).items():
            m = getattr(system, name, None)
            if m is not None:
                m.freeze() if frozen else m.unfreeze()

        return system
    
    def copy(self) -> "HoloSystem":
        """
        Return a deep copy of this HoloSystem, including all states, parameters and buffers.
        """
        return pickle.loads(pickle.dumps(self))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def report(self) -> plt.Figure:
        """Full-state figure of this system."""
        return self._render_report(other=None)

    def report_diff(self, other: "HoloSystem") -> plt.Figure:
        """
        Full-state figure comparing this system against another HoloSystem *other*.

        Same layout as report(), but every panel shows A - B rather than A:
        image panels are differenced and drawn on a diverging colormap
        centred on zero, the LUT is drawn as a difference curve, and the
        Seidel coefficients as difference bars. The camera panel is the one
        exception -- two frames on a shared axis say more than their
        difference would, so A and B are overlaid there.

        Only panels active on *self* are drawn (same rule as report()) --
        if the two systems have different modules enabled, mismatches will
        surface as an empty/omitted column rather than an error.
        """
        return self._render_report(other=other)


    # ------------------------------------------------------------------
    # Shared report plotting helpers
    # ------------------------------------------------------------------

    #: colormap used for every image panel in diff mode, whatever the panel
    #: shows in absolute mode. A difference is signed and centred on zero
    _DIFF_CMAP = "bwr"

    @staticmethod
    def _phase_diff(a: np.ndarray, b: np.ndarray,
                    weight_a: np.ndarray = None, weight_b: np.ndarray = None):
        """
        Signed phase difference (a - b) in (-pi, pi], with the global phase
        offset removed first. Returns (difference, removed_offset).

        A field and the same field times exp(i c) produce identical
        intensities, so c is a gauge, not a property of the fit. A
        raw wrapped subtraction reports it as error, which is what makes a
        recovered phase map look like a flat plane at some arbitrary value
        with a colorbar spanning radians: the panel is showing c, and the
        structure that actually matters is buried under it.

        The offset is the circular mean of the difference, weighted by the
        amplitudes when they are given. Weighting matters: where the field
        is near zero its phase is numerically arbitrary, and unweighted
        those pixels drag the mean off. The residual is wrapped afterwards,
        so a difference that straddles the branch cut still reads correctly.
        """
        d = a - b
        w = np.ones_like(d)
        if weight_a is not None:
            w = w * weight_a
        if weight_b is not None:
            w = w * weight_b
        total = w.sum()
        if total <= 0 or not np.isfinite(total):
            w, total = np.ones_like(d), float(d.size)
        offset = float(np.angle(np.sum(w * np.exp(1j * d)) / total))
        return (d - offset + np.pi) % (2 * np.pi) - np.pi, offset

    def _add_colorbar(self, fig, ax, im, *, label=None):
        div = make_axes_locatable(ax)
        cax = div.append_axes("bottom", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")

        vmin, vmax = im.get_clim()
        cbar.set_ticks([vmin, 0.5 * (vmin + vmax), vmax])

        scale = max(abs(vmin), abs(vmax))
        exponent = int(np.floor(np.log10(scale))) if scale > 0 else 0
        if abs(exponent) >= 3:
            factor = 10 ** exponent
            cbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/factor:.2f}"))
            suffix = rf"$\times 10^{{{exponent}}}$"
        else:
            cbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2f}"))
            suffix = ""

        if label:
            cbar.set_label(f"{label} {suffix}".rstrip())
        elif suffix:
            cbar.set_label(suffix)
        return cbar

    def _draw_image_panel(self, ax, interp, key: str, other: "HoloSystem" = None):
        """Generic renderer for every entry in _IMAGE_PANELS, in absolute or diff mode."""
        cfg = self._IMAGE_PANELS[key]
        data_a = cfg["data"](self)

        kwargs = {}
        if cfg.get("extent") is not None:
            kwargs["extent"] = cfg["extent"]
        if cfg.get("aspect") is not None:
            kwargs["aspect"] = cfg["aspect"]

        title = cfg.get("title")
        cmap = cfg["cmap"]
        note = None

        if other is None:
            data = data_a
        else:
            data_b = cfg["data"](other)
            if cfg["is_phase"]:
                weight = cfg.get("weight")
                data, offset = self._phase_diff(
                    data_a, data_b,
                    weight(self) if weight else None,
                    weight(other) if weight else None,
                )
                note = rf"global {offset:+.2f} rad removed"
            else:
                data = data_a - data_b
            # Symmetric about zero so the colour says the sign, not just
            # the magnitude, and so white always means "no difference".
            vmax = float(np.abs(data).max()) or 1e-12
            kwargs["vmin"], kwargs["vmax"] = -vmax, vmax
            cmap = self._DIFF_CMAP
            if title:
                title = f"Δ {title}"

        im = ax.imshow(data, cmap=cmap, interpolation=interp, **kwargs)
        if title:
            ax.set_title(title, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        if note:
            ax.text(0.02, 0.98, note, transform=ax.transAxes, ha="left", va="top",
                    fontsize=6, color="0.25",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0))
        label = cfg.get("label")
        if other is not None and label:
            label = f"Δ {label}"
        self._add_colorbar(ax.get_figure(), ax, im, label=label)

        annotate = cfg.get("annotate")
        if annotate is not None:
            annotate(ax, self, other)

        return im

    # ------------------------------------------------------------------
    # Panels that aren't image-shaped: drawn as difference curves / bars
    # ------------------------------------------------------------------

    def _draw_lut(self, ax, other: "HoloSystem" = None):
        g_res = 256
        g = torch.linspace(0, 1, g_res, device=self.geometry.device)
        with torch.no_grad():
            phi_a = self.lut.phase(g).cpu().numpy()
        g_np = g.cpu().numpy()

        if other is None:
            ax.plot(g_np, phi_a)
            ax.set_title("LUT", fontweight="bold")
            ax.set_ylabel("Phase [rad]")
        else:
            with torch.no_grad():
                phi_b = other.lut.phase(g.to(other.geometry.device)).cpu().numpy()
            ax.plot(g_np, phi_a - phi_b, color="C3")
            ax.axhline(0.0, color="0.6", lw=0.8, ls="--")
            ax.set_title("Δ LUT (A - B)", fontweight="bold")
            ax.set_ylabel("Δ Phase [rad]")
            peak = float(np.abs(phi_a - phi_b).max())
            ax.text(0.03, 0.95, rf"max |Δ| = {peak:.3g} rad", transform=ax.transAxes,
                    ha="left", va="top", fontsize=7, color="0.25")
        ax.set_xlabel("Grayscale")
        ax.set_box_aspect(1)

    def _draw_seidel(self, ax, other: "HoloSystem" = None):
        names = self.pupil.coeffs.NAMES if hasattr(self.pupil.coeffs, "NAMES") else \
            type(self).__module__
        names = self.pupil.seidel_coeffs.NAMES if hasattr(self.pupil, "seidel_coeffs") else names

        coeffs_a = self.pupil.coeffs.detach().cpu().numpy()
        x = np.arange(len(coeffs_a))

        ax_pos = ax.get_position()
        ax.set_position([ax_pos.x0 + 0.03, ax_pos.y0, ax_pos.width * 0.92, ax_pos.height])

        if other is None:
            ax.bar(x, coeffs_a)
            ax.set_ylabel("Value [a.u.]", labelpad=0)
            ax.set_title("Aberr. Coeffs", fontweight="bold")
        else:
            # Difference bars 
            coeffs_b = other.pupil.coeffs.detach().cpu().numpy()
            delta = coeffs_a - coeffs_b
            ax.bar(x, delta, color="C3")
            ax.axhline(0.0, color="0.6", lw=0.8, ls="--")
            ax.set_ylabel("Δ Value [a.u.]", labelpad=0)
            ax.set_title("Δ Aberr. Coeffs", fontweight="bold")
            ax.text(0.03, 0.95, rf"max |Δ| = {np.abs(delta).max():.3g}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=7,
                    color="0.25")

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_box_aspect(1)

        if self.pupil._use_vignetting:
            d = self.pupil.aperture_distance.item()
            offset = self.pupil.aperture_offset.detach().cpu().numpy()
            if other is None:
                text = (rf"$d$ = {d:.2f} mm" + "\n" +
                        rf"offset = ({offset[0]:.2f}, {offset[1]:.2f}) mm")
            else:
                d_b = other.pupil.aperture_distance.item()
                offset_b = other.pupil.aperture_offset.detach().cpu().numpy()
                text = (rf"Δ$d$ = {d - d_b:+.3f} mm" + "\n" +
                        rf"Δoffset = ({offset[0]-offset_b[0]:+.3f}, "
                        rf"{offset[1]-offset_b[1]:+.3f}) mm")
            ax.text(0.5, -0.55, text, transform=ax.transAxes, ha="center", va="top")

    def _draw_affine(self, ax, other: "HoloSystem" = None):
        corners_cam_a, corners_fov = self.camera.get_corners()
        c_fov = corners_fov.cpu().detach().numpy()
        c_cam_a = corners_cam_a.cpu().detach().numpy()

        ax.plot(c_fov[:, 0], c_fov[:, 1], "k-", label="Far-field FoV")
        ax.plot(c_cam_a[:, 0], c_cam_a[:, 1], "r-", label="Camera FoV (A)" if other else "Camera FoV")

        if other is not None:
            corners_cam_b, _ = other.camera.get_corners()
            c_cam_b = corners_cam_b.cpu().detach().numpy()
            ax.plot(c_cam_b[:, 0], c_cam_b[:, 1], "b--", label="Camera FoV (B)")

        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_title("Camera" if other is None else "Camera (A vs B)", fontweight="bold")
        ax.set_xlabel("u [norm]"); ax.set_ylabel("v [norm]", labelpad=-3)
        ax.set_box_aspect(1)
        ax.legend(fontsize=7, loc="lower center", framealpha=0.8)

        p = self.camera.affine_params
        if other is None:
            text = (rf"$\theta$ = {np.degrees(p['rotation']):.2f}°" + "\n" +
                    f"$s_x$ = {p['scale'][0]:.3f}, $s_y$ = {p['scale'][1]:.3f}\n" +
                    f"$t_x$ = {p['shift'][0]:.2f}, $t_y$ = {p['shift'][1]:.2f}")
        else:
            pb = other.camera.affine_params
            text = (rf"Δ$\theta$ = {np.degrees(p['rotation'] - pb['rotation']):+.3f}°" + "\n" +
                    f"Δ$s$ = ({p['scale'][0]-pb['scale'][0]:+.4f}, "
                    f"{p['scale'][1]-pb['scale'][1]:+.4f})\n" +
                    f"Δ$t$ = ({p['shift'][0]-pb['shift'][0]:+.2f}, "
                    f"{p['shift'][1]-pb['shift'][1]:+.2f}) px")
        # Camera pixel count / FoV sample count are geometry properties
        # — shown once from self's geometry.
        text += ("\n\n" +
                 rf"SLM: {self.geometry.slm_pixels[1]}$\times${self.geometry.slm_pixels[0]} pixels" + "\n" +
                 rf"Camera: {self.geometry.camera_shape[1]}$\times${self.geometry.camera_shape[0]} pixels" + "\n" +
                 rf"FoV: {self.geometry.far_fov_samples[1]}$\times${self.geometry.far_fov_samples[0]} samples"
                 )
        ax.text(0.5, -0.45, text, transform=ax.transAxes, ha="center", va="top")

    # ------------------------------------------------------------------
    # Annotation helpers for image panels
    # ------------------------------------------------------------------

    def _annotate_crosstalk(ax, s, other=None):
        K = s.pixel._pixel_samples
        Kp = s.pixel._kernel_pixels
        c = (K * Kp) // 2
        for fn in (ax.axvline, ax.axhline):
            fn(c - K / 2, color="cyan", linestyle="--", lw=1)
            fn(c + K / 2, color="cyan", linestyle="--", lw=1)
        ax.plot([], [], color="cyan", linestyle="--", lw=1, label="Pixel Edges")
        ax.legend(fontsize=8, loc="lower center", framealpha=0.8)

        residual_suffix = "\n+ residual" if s.pixel._use_residual else ""
        sig_a = s.pixel.sigma.detach().cpu().numpy()
        if other is None:
            text = rf"$\sigma_x$ = {sig_a[1]:.2f}, $\sigma_y$ = {sig_a[0]:.2f}" + residual_suffix
        else:
            sig_b = other.pixel.sigma.detach().cpu().numpy()
            text = (rf"A: $\sigma_x$={sig_a[1]:.3f}, $\sigma_y$={sig_a[0]:.3f}" + "\n" +
                    rf"Δ: {sig_a[1]-sig_b[1]:+.3f}, {sig_a[0]-sig_b[0]:+.3f}" + residual_suffix)
        ax.text(0.5, -0.55, text, transform=ax.transAxes, ha="center", va="top")

    def _annotate_pixel_envelope(ax, s, other=None):
        fill_a = s.pixel.fill
        rho_a = s.pixel.deadspace_reflectance.item()

        if other is None:
            text = rf"$F$ = {fill_a:.3f},  $\rho$ = {rho_a:+.3f}"
        else:
            rho_b = other.pixel.deadspace_reflectance.item()
            text = (rf"A: $F$={fill_a:.3f}, $\rho$={rho_a:+.3f}" + "\n" +
                    rf"Δ$F$={fill_a-other.pixel.fill:+.4f}, Δ$\rho$={rho_a-rho_b:+.4f}")
        ax.text(0.5, -0.55, text, transform=ax.transAxes, ha="center", va="top")

    # =============================================================================
    # Declarative registry of every image-shaped panel.
    # Each "data" getter takes a HoloSystem and returns a plain numpy array.
    # "weight", where present, returns the amplitude that the corresponding
    # phase should be weighted by when a global phase offset is estimated --
    # see _phase_diff(). Without it, pixels where the field is essentially
    # zero (most of the far-field background, for instance) contribute
    # arbitrary phases to that estimate with equal weight.
    # =============================================================================

    _IMAGE_PANELS = {
        "slm_amp": dict(
            data=lambda s: s.slm_field.field.abs().cpu().detach().numpy(),
            cmap="hot", label="Amplitude [a.u.]", title="SLM Field", is_phase=False,
        ),
        "slm_phase": dict(
            data=lambda s: s.slm_field.field.angle().cpu().detach().numpy(),
            weight=lambda s: s.slm_field.field.abs().cpu().detach().numpy(),
            cmap="twilight", label="Phase [rad]", title=None, is_phase=True,
        ),
        "depth_map": dict(
            data=lambda s: s.lut.depth_map.cpu().detach().numpy(),
            cmap="bone", label="Relative depth [a.u.]", title="Cell Gap", is_phase=False,
        ),
        "crosstalk": dict(
            data=lambda s: s.pixel.kernel.detach().cpu().numpy(),
            cmap="hot", label="Magnitude [a.u.]", title="Crosstalk Kernel", is_phase=False,
            annotate=_annotate_crosstalk,
        ),
        "pixel_envelope": dict(
            data=lambda s: s.pixel.aperture_envelope.detach().abs().square().cpu().numpy(),
            cmap="gray", label="Normalised intensity [a.u.]", title="Pixel envelope",
            is_phase=False, extent=(0, 1, 0, 1), aspect="equal",
            annotate=_annotate_pixel_envelope,
        ),
        "back_amp_slm": dict(
            data=lambda s: s.background.get_filtered(mode="slm").abs().cpu().detach().numpy(),
            cmap="hot", label="Amplitude [a.u.]", title="SLM Aperture", is_phase=False,
        ),
        "back_phase_slm": dict(
            data=lambda s: s.background.get_filtered(mode="slm").angle().cpu().detach().numpy(),
            weight=lambda s: s.background.get_filtered(mode="slm").abs().cpu().detach().numpy(),
            cmap="twilight", label="Phase [rad]", title=None, is_phase=True,
        ),
        "back_amp_far": dict(
            data=lambda s: s.background.get_filtered(mode="far").abs().cpu().detach().numpy(),
            cmap="hot", label="Amplitude [a.u.]", title="Reflections",
            is_phase=False, extent=(0, 1, 0, 1), aspect="equal",
        ),
        "back_phase_far": dict(
            data=lambda s: s.background.get_filtered(mode="far").angle().cpu().detach().numpy(),
            weight=lambda s: s.background.get_filtered(mode="far").abs().cpu().detach().numpy(),
            cmap="twilight", label="Phase [rad]", title=None,
            is_phase=True, extent=(0, 1, 0, 1), aspect="equal",
        ),
    }

    # ------------------------------------------------------------------
    # Column layout — identical logic for report() and report_diff();
    # only which system(s) feed the drawers differs.
    # ------------------------------------------------------------------

    def _build_columns(self, other: "HoloSystem" = None):
        columns = []

        if self.lut.is_active:
            if self.lut.use_depth:
                columns.append((
                    lambda ax: self._draw_lut(ax, other),
                    lambda ax: self._draw_image_panel(ax, "bilinear", "depth_map", other),
                ))
            else:
                columns.append((lambda ax: self._draw_lut(ax, other),))

        if self.slm_field.is_active:
            columns.append((
                lambda ax: self._draw_image_panel(ax, "bilinear", "slm_amp", other),
                lambda ax: self._draw_image_panel(ax, "bilinear", "slm_phase", other),
            ))

        if self.background.is_active:
            columns.append((
                lambda ax: self._draw_image_panel(ax, "bilinear", "back_amp_slm", other),
                lambda ax: self._draw_image_panel(ax, "bilinear", "back_phase_slm", other),
            ))
            columns.append((
                lambda ax: self._draw_image_panel(ax, "bilinear", "back_amp_far", other),
                lambda ax: self._draw_image_panel(ax, "bilinear", "back_phase_far", other),
            ))

        if self.pupil.is_active:
            columns.append((lambda ax: self._draw_seidel(ax, other),))

        if self.pixel.is_active:
            columns.append((lambda ax: self._draw_image_panel(ax, "nearest", "crosstalk", other),))
            columns.append((
                lambda ax: self._draw_image_panel(ax, "bilinear", "pixel_envelope", other),
            ))

        if self.camera.is_active:
            columns.append((lambda ax: self._draw_affine(ax, other),))

        return columns

    def _render_report(self, other: "HoloSystem" = None) -> plt.Figure:
        columns = self._build_columns(other)
        n_cols = len(columns)

        if n_cols == 0:
            fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
            ax.text(0.5, 0.5, "No active modules", ha="center", va="center", fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])
            return fig

        col_w, row_h, n_rows = 2.0, 2.2, 2
        fig = plt.figure(figsize=(col_w * n_cols, row_h * n_rows), dpi=100)
        gs = fig.add_gridspec(
            n_rows, n_cols, hspace=0.55, wspace=0.1,
            left=0.04, right=0.97, top=0.8, bottom=0.08,
        )
        if other is not None:
            fig.suptitle("A - B discrepancy", fontsize=10, y=0.97)

        for col_idx, fns in enumerate(columns):
            if len(fns) == 1:
                ax = fig.add_subplot(gs[0, col_idx])
                fns[0](ax)
            else:
                ax_top = fig.add_subplot(gs[0, col_idx])
                ax_bot = fig.add_subplot(gs[1, col_idx])
                fns[0](ax_top)
                fns[1](ax_bot)

        return fig
