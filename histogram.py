import numpy as np
import torch
import torch.nn as nn


class HistLayer(nn.Module):
    def __init__(self, in_channels, num_bins=4, dct=False, two_d=False):

        # inherit nn.module
        super(HistLayer, self).__init__()

        # define layer properties
        # histogram bin data
        self.in_channels = in_channels
        self.numBins = num_bins
        self.learnable = False
        bin_edges = np.linspace(-1, 1, num_bins + 1)
        centers = bin_edges + (bin_edges[2] - bin_edges[1]) / 2
        self.centers = centers[:-1]
        self.width = (bin_edges[2] - bin_edges[1]) / 2
        self.dct = dct
        self.two_d = two_d

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

    def forward(self, xx):
        if self.dct:
            xx = torch_dct(xx)
        xx = self.bin_centers_conv(xx)
        xx = torch.abs(xx)
        xx = self.bin_widths_conv(xx)
        xx = torch.pow(torch.empty_like(xx).fill_(1.01), xx)
        xx = self.threshold(xx)
        two_d = torch.flatten(xx, 2)
        xx = self.hist_pool(xx)
        one_d = torch.flatten(xx, 1)
        return one_d, two_d


# _hist_layer = HistLayer(in_channels=1, num_bins=256).to(device)


def emd_loss(hgram1, hgram2):
    return (
        ((torch.cumsum(hgram1, dim=1) - torch.cumsum(hgram2, dim=1)) ** 2)
        .sum(1)
        .mean(-1)
        .mean()
    )


def mae_loss(y, y_pred):
    return (torch.abs(y - y_pred)).sum(1).mean(-1).mean()


def mse_loss(y, y_pred):
    return torch.pow(y - y_pred, 2).sum(1).mean(-1).mean()


def extract_hist(layer, input):
    """
    return tuple of (one_d, two_d) histogram for each channel
    """
    _, num_ch, _, _ = input.shape
    hists = []
    for ch in range(num_ch):
        hists.append(layer(input[:, ch, :, :].unsqueeze(1)))
    #     hists = torch.stack(hists, 1)
    return hists


def extract_1d_hist(layer, input):
    """
    return tuple of (one_d, two_d) histogram for each channel
    """
    _, num_ch, _, _ = input.shape
    hists = []
    for ch in range(num_ch):
        hists.append(layer(input[:, ch, :, :].unsqueeze(1))[0])
    hists = torch.stack(hists, 1)
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
    """
    Joint entropy of two histograms
    """
    pxy = torch.bmm(hgram1, hgram2.transpose(1, 2)) / hgram1.shape[-1]
    nz = pxy > 0
    return -torch.sum(pxy[nz] * torch.log(pxy[nz]))


def dmi(hgram1, hgram2):
    return 1 - (mutual_information(hgram1, hgram2) / entropy(hgram1, hgram2))


def histogram_losses(hgram1, hgram2):
    """
    compute histogram losses (EMD and MI) for each channel
    hgram1 and hgram2 are tuples of (one_d, two_d) histograms for each channel
    """
    emd = 0
    mi = 0
    for channel_hgram1, channel_hgram2 in zip(hgram1, hgram2):
        emd += emd_loss(channel_hgram1[0], channel_hgram2[0])
        mi += dmi(channel_hgram1[1], channel_hgram2[1])
    return emd / 3, mi / 3
