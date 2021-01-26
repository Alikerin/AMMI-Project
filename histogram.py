from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class HistLayer(nn.Module):
    """Deep Neural Network Layer for Computing Differentiable Histogram.

    Computes a differentiable histogram using a hard-binning operation implemented using
    CNN layers as desribed in `"Differentiable Histogram with Hard-Binning"
    <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Attributes:
        in_channel (int): Number of image input channels.
        numBins (int): Number of histogram bins.
        learnable (bool): Flag to determine whether histogram bin widths and centers are
            learnable.
        centers (List[float]): Histogram centers.
        widths (List[float]): Histogram widths.
        two_d (bool): Flag to return flattened or 2D histogram.
        bin_centers_conv (nn.Module): 2D CNN layer with weight=1 and bias=`centers`.
        bin_widths_conv (nn.Module): 2D CNN layer with weight=-1 and bias=`width`.
        threshold (nn.Module): DNN layer for performing hard-binning.
        hist_pool (nn.Module): Pooling layer.
    """

    def __init__(self, in_channels, num_bins=4, two_d=False):
        super(HistLayer, self).__init__()

        # histogram data
        self.in_channels = in_channels
        self.numBins = num_bins
        self.learnable = False
        bin_edges = np.linspace(-1.05, 1.05, num_bins + 1)
        centers = bin_edges + (bin_edges[2] - bin_edges[1]) / 2
        self.centers = centers[:-1]
        self.width = (bin_edges[2] - bin_edges[1]) / 2
        self.two_d = two_d

        # prepare NN layers for histogram computation
        self.bin_centers_conv = nn.Conv2d(
            self.in_channels,
            self.numBins * self.in_channels,
            1,
            groups=self.in_channels,
            bias=True,
        )
        self.bin_centers_conv.weight.data.fill_(1)
        self.bin_centers_conv.weight.requires_grad = False
        self.bin_centers_conv.bias.data = torch.nn.Parameter(
            -torch.tensor(self.centers, dtype=torch.float32)
        )
        self.bin_centers_conv.bias.requires_grad = self.learnable

        self.bin_widths_conv = nn.Conv2d(
            self.numBins * self.in_channels,
            self.numBins * self.in_channels,
            1,
            groups=self.numBins * self.in_channels,
            bias=True,
        )
        self.bin_widths_conv.weight.data.fill_(-1)
        self.bin_widths_conv.weight.requires_grad = False
        self.bin_widths_conv.bias.data.fill_(self.width)
        self.bin_widths_conv.bias.requires_grad = self.learnable

        self.centers = self.bin_centers_conv.bias
        self.widths = self.bin_widths_conv.weight
        self.threshold = nn.Threshold(1, 0)
        self.hist_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, input_image):
        """Computes differentiable histogram.

        Args:
            input_image: input image.

        Returns:
            flattened and un-flattened histogram.
        """
        # |x_i - u_k|
        xx = self.bin_centers_conv(input_image)
        xx = torch.abs(xx)

        # w_k - |x_i - u_k|
        xx = self.bin_widths_conv(xx)

        # 1.01^(w_k - |x_i - u_k|)
        xx = torch.pow(torch.empty_like(xx).fill_(1.01), xx)

        # Φ(1.01^(w_k - |x_i - u_k|), 1, 0)
        xx = self.threshold(xx)

        # clean-up
        two_d = torch.flatten(xx, 2)
        xx = self.hist_pool(xx)  # xx.sum([2, 3])
        one_d = torch.flatten(xx, 1)
        return one_d, two_d


def emd_loss(hgram1, hgram2):
    """Computes Earth Mover's Distance (EMD) between histograms

    Args:
        histogram_1: first histogram tensor, shape: batch_size x num_bins.
        histogram_1: second histogram tensor, shape: batch_size x num_bins.

    Returns:
        EMD loss.
    """
    return (
        ((torch.cumsum(hgram1, dim=1) - torch.cumsum(hgram2, dim=1)) ** 2)
        .sum(1)
        .mean(-1)
        .mean()
    )


def mae_loss(histogram_1: Tensor, histogram_2: Tensor) -> Tensor:
    """Computes Mean Absolute Error (MAE) between histograms

    Args:
        histogram_1: first histogram tensor, shape: batch_size x num_bins.
        histogram_1: second histogram tensor, shape: batch_size x num_bins.

    Returns:
        MAE loss.
    """
    return (torch.abs(histogram_1 - histogram_2)).sum(1).mean(-1).mean()


def mse_loss(histogram_1: Tensor, histogram_2: Tensor) -> Tensor:
    """Computes Mean Squared Error (MSE) between histograms.

    Args:
        histogram_1: first histogram tensor, shape: batch_size x num_bins.
        histogram_1: second histogram tensor, shape: batch_size x num_bins.

    Returns:
        MSE loss.
    """
    return torch.pow(histogram_1 - histogram_2, 2).sum(1).mean(-1).mean()


def extract_hist(
    layer: HistLayer, image: Tensor, one_d=False, color_space="RGB"
) -> List[Tuple[Tensor, Tensor]]:
    """Extracts both vector and 2D histogram.

    Args:
        layer: histogram layer.
        image: input image tensor, shape: batch_size x num_channels x width x height.

    Returns:
        list of tuples containing 1d (and 2d histograms) for each channel.
        1d histogram shape: batch_size x num_bins
        2d histogram shape: batch_size x num_bins x width*height
    """
    _, num_ch, _, _ = image.shape
    # convert to desired color space
    if color_space == "YUV":
        image = rgb2yuv(image)
    hists = []
    for ch in range(num_ch):
        hists.append(layer(image[:, ch, :, :].unsqueeze(1)))
    if one_d:
        return [one_d_hist for (one_d_hist, _) in hists]
    return hists


def mutual_information(hgram1, hgram2):
    """Mutual information for joint histogram"""
    pxy = torch.bmm(hgram1, hgram2.transpose(1, 2)) / hgram1.shape[-1]
    pxy += 1e-6
    px = torch.sum(pxy, axis=1)  # marginal for x over y
    py = torch.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    s = pxy[nzs] * torch.log(pxy[nzs] / px_py[nzs])
    #     s = pxy * torch.log(pxy / px_py)
    return torch.sum(s)


def entropy(hgram1, hgram2):
    """Joint entropy of two histograms"""
    pxy = torch.bmm(hgram1, hgram2.transpose(1, 2)) / hgram1.shape[-1]
    nz = pxy > 0
    return -torch.sum(pxy[nz] * torch.log(pxy[nz]))


def dmi(hgram1, hgram2):
    return 1 - (mutual_information(hgram1, hgram2) / entropy(hgram1, hgram2))


def histogram_losses(
    histogram_1: List[Tensor], histogram_2: List[Tensor], loss_type: str = "emd"
) -> Tuple[float, float]:
    """Compute Histogram Losses.

    Computes EMD and MI losses for each channel, then returns the mean.

    Args:
        histogram_1: first histogram tensor, shape: batch_size x num_channels x num_bins.
        histogram_1: second histogram tensor, shape: batch_size x num_channels x num_bins
        loss_type: type of loss function.

    Returns:
        Tuple containing mean of EMD and MI losses respectively.
    """
    emd = 0
    mi = 0
    if loss_type == "emd":
        loss_fn = emd_loss
    elif loss_type == "mae":
        loss_fn = mae_loss
    else:
        loss_fn = mse_loss
    for channel_hgram1, channel_hgram2 in zip(histogram_1, histogram_2):
        emd += loss_fn(channel_hgram1[0], channel_hgram2[0])
        # mi += dmi(channel_hgram1[1], channel_hgram2[1])
    return emd / 3, mi / 3


def rgb2yuv(image: Tensor):
    """Converts image from RGB to YUV color space.

    Arguments:
        image: batch of images with shape (batch_size x num_channels x width x height).

    Returns:
        batch of images in YUV color space with shape
        (batch_size x num_channels x width x height).
    """
    y = (
        (0.299 * image[:, 0, :, :])
        + (0.587 * image[:, 1, :, :])
        + (0.114 * image[:, 2, :, :])
    )
    u = (
        (-0.14713 * image[:, 0, :, :])
        + (-0.28886 * image[:, 1, :, :])
        + (0.436 * image[:, 2, :, :])
    )
    v = (
        (0.615 * image[:, 0, :, :])
        + (-0.51499 * image[:, 1, :, :])
        + (-0.10001 * image[:, 2, :, :])
    )
    image = torch.stack([y, u, v], 1)
    return image
