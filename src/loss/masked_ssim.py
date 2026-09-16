"""
Masked SSIM and Multi-Scale Masked SSIM loss functions.
These follow the conventions of the pytorch-msssim package, 
but support a binary mask that excludes certain regions from the SSIM computation. 
This is useful for applications where certain areas of the image should not contribute to the loss, 
such as when there are occlusions or irrelevant regions in the images being compared.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# A separable Gaussian filter, which is used in the MaskedSSIM and MaskedMS_SSIM classes. 
# The filter is applied separately in the horizontal and vertical directions, 
# which is more efficient than applying a 2D Gaussian filter.
class SeparableGaussianFilter(nn.Module):

    def __init__(self, 
                 window_size: int = 11, 
                 sigma: float = 1.5, 
                 device: str = "cpu"
    ):
        super().__init__()

        coords = torch.arange(window_size, dtype=torch.float32, device=device)
        coords -= window_size // 2

        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()

        self.register_buffer(
            "kernel_h",
            g.view(1, 1, 1, window_size)
        )

        self.register_buffer(
            "kernel_v",
            g.view(1, 1, window_size, 1)
        )

    def forward(self, x):

        C = x.shape[1]

        #
        # EXACTLY like pytorch-msssim:
        # valid convolutions, no padding
        #

        x = F.conv2d(
            x,
            self.kernel_h.expand(C, 1, 1, -1),
            groups=C,
            padding=0,
        )

        x = F.conv2d(
            x,
            self.kernel_v.expand(C, 1, -1, 1),
            groups=C,
            padding=0,
        )

        return x

# Masked SSIM. The SSIM is computed only over the region where mask=1, which means that 
# it avoids including edges of the mask which would occur if input images were multiplied 
# by the mask and then fed to a standard SSIM, which could dominate the SSIM function.
class MaskedSSIM(nn.Module):

    def __init__(
        self,
        mask: torch.Tensor,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        K: tuple = (0.01, 0.03),
        eps: float = 1e-12,
    ):
        super().__init__()

        self.eps = eps

        K1, K2 = K

        self.C1 = (K1 * data_range) ** 2
        self.C2 = (K2 * data_range) ** 2

        self.filter = SeparableGaussianFilter(
            window_size,
            sigma,
            device=mask.device,
        )

        self.register_buffer("M", None)
        self.register_buffer("Wm", None)

        self.set_mask(mask)

    def set_mask(self, mask):

        if mask.ndim == 2:

            M = mask[None, None]

        elif mask.ndim == 3:

            M = mask[:, None]

        elif mask.ndim == 4:

            M = mask

        else:

            raise ValueError(
                "mask must have shape (H,W), (N,H,W), or (N,1,H,W)"
            )

        M = M.float()

        Wm = self.filter(M)

        self.M = M
        self.Wm = Wm

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ) -> torch.Tensor:

        if X.shape != Y.shape:

            raise ValueError(
                f"Input images should have the same dimensions, "
                f"but got {X.shape} and {Y.shape}."
            )

        if X.ndim != 4:

            raise ValueError(
                f"Input images should be 4D tensors, but got {X.shape}"
            )

        valid_map = (self.Wm > self.eps).float()

        XM = X * self.M
        YM = Y * self.M

        XXM = X * XM
        YYM = Y * YM
        XYM = X * YM

        mu1 = self.filter(XM) / (self.Wm + self.eps)
        mu2 = self.filter(YM) / (self.Wm + self.eps)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            self.filter(XXM) / (self.Wm + self.eps)
            - mu1_sq
        )

        sigma2_sq = (
            self.filter(YYM) / (self.Wm + self.eps)
            - mu2_sq
        )

        sigma12 = (
            self.filter(XYM) / (self.Wm + self.eps)
            - mu1_mu2
        )

        cs_map = (
            (2 * sigma12 + self.C2)
            / (sigma1_sq + sigma2_sq + self.C2)
        )

        ssim_map = (
            (2 * mu1_mu2 + self.C1)
            / (mu1_sq + mu2_sq + self.C1)
        ) * cs_map

        denom = (
            valid_map.flatten(1)
            .sum(-1)
            .clamp_min(self.eps)
        )

        ssim_per_channel = (
            (ssim_map * valid_map)
            .flatten(1)
            .sum(-1)
            / denom
        )

        return ssim_per_channel.mean()
    
# Multi-scale version of MaskedSSIM
class MaskedMS_SSIM(nn.Module):

    def __init__(
        self,
        mask: torch.Tensor,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        weights: Optional[list] = None,
        K=(0.01, 0.03),
        eps: float = 1e-12,
    ):
        super().__init__()

        self.eps = eps

        K1, K2 = K

        self.C1 = (K1 * data_range) ** 2
        self.C2 = (K2 * data_range) ** 2

        self.filter = SeparableGaussianFilter(
            window_size,
            sigma,
            device=mask.device,
        )

        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

        self.register_buffer(
            "weights",
            torch.tensor(weights, dtype=torch.float32, device=mask.device),
        )

        self.levels = len(weights)

        self.masks = []
        self.Wms = []

        self.set_mask(mask)

    def set_mask(self, mask):

        if mask.ndim == 2:

            M = mask[None, None]

        elif mask.ndim == 3:

            M = mask[:, None]

        elif mask.ndim == 4:

            M = mask

        else:

            raise ValueError(
                "mask must have shape (H,W), (N,H,W), or (N,1,H,W)"
            )

        M = M.float()

        #
        # Remove previous pyramid buffers if set_mask()
        # is called more than once.
        #

        for level in range(self.levels):

            for name in (
                f"mask_{level}",
                f"Wm_{level}",
            ):

                if name in self._buffers:

                    del self._buffers[name]

        #
        # Build pyramid
        #

        for level in range(self.levels):

            Wm = self.filter(M)

            self.register_buffer(
                f"mask_{level}",
                M,
            )

            self.register_buffer(
                f"Wm_{level}",
                Wm,
            )

            if level < self.levels - 1:

                padding = [
                    s % 2
                    for s in M.shape[-2:]
                ]

                M = F.avg_pool2d(
                    M,
                    kernel_size=2,
                    padding=padding,
                )

    def _masked_ssim(
        self,
        X,
        Y,
        M,
        Wm,
    ):

        valid_map = (Wm > self.eps).float()

        XM = X * M
        YM = Y * M

        XXM = X * XM
        YYM = Y * YM
        XYM = X * YM

        mu1 = self.filter(XM) / (Wm + self.eps)
        mu2 = self.filter(YM) / (Wm + self.eps)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            self.filter(XXM) / (Wm + self.eps)
            - mu1_sq
        )

        sigma2_sq = (
            self.filter(YYM) / (Wm + self.eps)
            - mu2_sq
        )

        sigma12 = (
            self.filter(XYM) / (Wm + self.eps)
            - mu1_mu2
        )

        cs_map = (
            (2 * sigma12 + self.C2)
            / (sigma1_sq + sigma2_sq + self.C2)
        )

        ssim_map = (
            (2 * mu1_mu2 + self.C1)
            / (mu1_sq + mu2_sq + self.C1)
        ) * cs_map

        denom = (
            valid_map.flatten(1)
            .sum(-1)
            .clamp_min(self.eps)
        )

        ssim_per_channel = (
            (ssim_map * valid_map)
            .flatten(1)
            .sum(-1)
            / denom
        )

        cs = (
            (cs_map * valid_map)
            .flatten(1)
            .sum(-1)
            / denom
        )

        return ssim_per_channel, cs

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ) -> torch.Tensor:

        if X.shape != Y.shape:

            raise ValueError(
                f"Input images should have the same dimensions, "
                f"but got {X.shape} and {Y.shape}."
            )

        if X.ndim != 4:

            raise ValueError(
                f"Input images should be 4D tensors, but got {X.shape}"
            )

        smaller_side = min(X.shape[-2:])

        required = (
            self.filter.kernel_h.shape[-1] - 1
        ) * (2 ** (len(self.weights) - 1))

        if smaller_side <= required:

            raise ValueError(
                f"Image size should be larger than {required}"
            )

        mcs = []

        for level in range(self.levels):

            M = getattr(
                self,
                f"mask_{level}",
            )

            Wm = getattr(
                self,
                f"Wm_{level}",
            )

            ssim_per_channel, cs = self._masked_ssim(
                X,
                Y,
                M,
                Wm,
            )

            if level < len(self.weights) - 1:

                mcs.append(torch.relu(cs))

                padding = [
                    s % 2
                    for s in X.shape[-2:]
                ]

                X = F.avg_pool2d(
                    X,
                    kernel_size=2,
                    padding=padding,
                )

                Y = F.avg_pool2d(
                    Y,
                    kernel_size=2,
                    padding=padding,
                )

        ssim_per_channel = torch.relu(
            ssim_per_channel
        )

        mcs_and_ssim = torch.stack(
            mcs + [ssim_per_channel],
            dim=0,
        )

        ms_ssim = torch.prod(
            mcs_and_ssim
            ** self.weights.view(-1, 1),
            dim=0,
        )

        return ms_ssim.mean()