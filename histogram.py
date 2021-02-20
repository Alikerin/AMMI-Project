from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        bin_edges = np.linspace(-0.05, 1.05, num_bins + 1)
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
        (torch.cumsum(hgram1, dim=1) - torch.cumsum(hgram2, dim=1))
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
    layer: HistLayer, image: Tensor, one_d: bool = False
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
    # convert from [-1, 1] to [0, 1]
    image = (image + 1) / 2
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


class HistogramLoss(nn.Module):
    def __init__(self, loss_fn, num_bins, rgb=True, yuv=True):
        super().__init__()
        self.rgb = rgb
        self.yuv = yuv
        self.histlayer = HistLayer(in_channels=1, num_bins=num_bins)
        loss_dict = {"emd": emd_loss, "mae": mae_loss, "mse": mse_loss}
        self.loss_fn = loss_dict[loss_fn]

    def to_YUV(self, image):
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

    def extract_hist(self, image, one_d=False):
        """Extracts both vector and 2D histogram.

        Args:
            layer: histogram layer.
            image: input image tensor, shape: batch_size x num_channels x width x height.

        Returns:
            list of tuples containing 1d (and 2d histograms) for each channel.
            1d histogram shape: batch_size x num_bins
            2d histogram shape: batch_size x num_bins x width*height
        """
        # comment next line if image is in [0,1] range
        image = (image + 1) / 2
        _, num_ch, _, _ = image.shape
        hists = []
        for ch in range(num_ch):
            hists.append(self.histlayer(image[:, ch, :, :].unsqueeze(1)))
        if one_d:
            return [one_d_hist for (one_d_hist, _) in hists]
        return hists

    def hist_loss(self, histogram_1, histogram_2):
        loss = 0
        for channel_hgram1, channel_hgram2 in zip(histogram_1, histogram_2):
            loss += self.loss_fn(channel_hgram1[0], channel_hgram2[0])
        return loss

    def __call__(self, input, reference):
        total_loss = 0
        if self.rgb:
            total_loss += self.hist_loss(
                self.extract_hist(input), self.extract_hist(reference)
            )
        if self.yuv:
            input_yuv = self.to_YUV(input)
            reference_yuv = self.to_YUV(reference)
            total_loss += self.hist_loss(
                self.extract_hist(input_yuv), self.extract_hist(reference_yuv)
            )
        return total_loss


class GPLoss(nn.Module):
    def __init__(self):
        super(GPLoss, self).__init__()
        self.trace = SPLoss()

    def get_image_gradients(self, input):
        f_v_1 = F.pad(input, (0, -1, 0, 0))
        f_v_2 = F.pad(input, (-1, 0, 0, 0))
        f_v = f_v_1 - f_v_2

        f_h_1 = F.pad(input, (0, 0, 0, -1))
        f_h_2 = F.pad(input, (0, 0, -1, 0))
        f_h = f_h_1 - f_h_2

        return f_v, f_h

    def __call__(self, input, reference):
        # comment these lines when you inputs and outputs are in [0,1] range already
        input = (input + 1) / 2
        reference = (reference + 1) / 2

        input_v, input_h = self.get_image_gradients(input)
        ref_v, ref_h = self.get_image_gradients(reference)

        trace_v = self.trace(input_v, ref_v)
        trace_h = self.trace(input_h, ref_h)
        return trace_v + trace_h


class CPLoss(nn.Module):
    def __init__(
        self, rgb=True, yuv=True, yuvgrad=True,
    ):
        super(CPLoss, self).__init__()
        self.rgb = rgb
        self.yuv = yuv
        self.yuvgrad = yuvgrad
        self.trace = SPLoss()
        self.trace_YUV = SPLoss()
        self.histlayer = HistLayer(in_channels=1, num_bins=256)

    def get_image_gradients(self, input):
        f_v_1 = F.pad(input, (0, -1, 0, 0))
        f_v_2 = F.pad(input, (-1, 0, 0, 0))
        f_v = f_v_1 - f_v_2

        f_h_1 = F.pad(input, (0, 0, 0, -1))
        f_h_2 = F.pad(input, (0, 0, -1, 0))
        f_h = f_h_1 - f_h_2

        return f_v, f_h

    def to_YUV(self, input):
        return torch.cat(
            (
                0.299 * input[:, 0, :, :].unsqueeze(1)
                + 0.587 * input[:, 1, :, :].unsqueeze(1)
                + 0.114 * input[:, 2, :, :].unsqueeze(1),
                0.493
                * (
                    input[:, 2, :, :].unsqueeze(1)
                    - (
                        0.299 * input[:, 0, :, :].unsqueeze(1)
                        + 0.587 * input[:, 1, :, :].unsqueeze(1)
                        + 0.114 * input[:, 2, :, :].unsqueeze(1)
                    )
                ),
                0.877
                * (
                    input[:, 0, :, :].unsqueeze(1)
                    - (
                        0.299 * input[:, 0, :, :].unsqueeze(1)
                        + 0.587 * input[:, 1, :, :].unsqueeze(1)
                        + 0.114 * input[:, 2, :, :].unsqueeze(1)
                    )
                ),
            ),
            dim=1,
        )

    def extract_hist(self, image, one_d=False):
        """Extracts both vector and 2D histogram.

        Args:
            layer: histogram layer.
            image: input image tensor, shape: batch_size x num_channels x width x height.

        Returns:
            list of tuples containing 1d (and 2d histograms) for each channel.
            1d histogram shape: batch_size x num_bins
            2d histogram shape: batch_size x num_bins x width*height
        """
        # comment next line if image is in [0,1] range
        image = (image + 1) / 2
        _, num_ch, _, _ = image.shape
        hists = []
        for ch in range(num_ch):
            hists.append(self.histlayer(image[:, ch, :, :].unsqueeze(1)))
        if one_d:
            return [one_d_hist for (one_d_hist, _) in hists]
        return hists

    def __call__(self, input, reference):
        # comment these lines when you inputs and outputs are in [0,1] range already
        input = (input + 1) / 2
        reference = (reference + 1) / 2
        total_loss = 0
        if self.rgb:
            total_loss += self.trace(input, reference)
        if self.yuv:
            input_yuv = self.to_YUV(input)
            reference_yuv = self.to_YUV(reference)
            total_loss += self.trace(input_yuv, reference_yuv)
        if self.yuvgrad:
            input_v, input_h = self.get_image_gradients(input_yuv)
            ref_v, ref_h = self.get_image_gradients(reference_yuv)

            total_loss += self.trace(input_v, ref_v)
            total_loss += self.trace(input_h, ref_h)

        return total_loss


class SPLoss(nn.Module):
    def __init__(self):
        super(SPLoss, self).__init__()

    def __call__(self, input, reference):
        a = torch.sum(
            torch.sum(
                F.normalize(input, p=2, dim=2) * F.normalize(reference, p=2, dim=2),
                dim=2,
                keepdim=True,
            )
        )
        b = torch.sum(
            torch.sum(
                F.normalize(input, p=2, dim=3) * F.normalize(reference, p=2, dim=3),
                dim=3,
                keepdim=True,
            )
        )
        return -(a + b) / input.size(2)
