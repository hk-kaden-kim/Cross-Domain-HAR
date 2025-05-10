import numpy as np
import torch
import torch.nn as nn
# from torch.autograd import Variable
# from typing import Union, List, Dict, Any, cast
import torch.nn.functional as F

# Source from: https://github.com/OxWearables/ssl-wearables
# modifided by Hyeongkyun Kim(hyeongkyun.kim@uzh.ch) and Orestis Oikonomou(orestis.oikonomou@uzh.ch)
# ---------------------------------------------------------------
# Copyright © 2022, University of Oxford
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
# A copy of the license is available at resources/license_mtssl

import scipy.stats as stats
import scipy.signal as signal

# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

class EvaClassifier(nn.Module):
    def __init__(self, input_size=1024, nn_size=512, output_size=2):
        super(EvaClassifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, nn_size)
        self.linear2 = torch.nn.Linear(nn_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

class Downsample(nn.Module):
    r"""Downsampling layer that applies anti-aliasing filters.
    For example, order=0 corresponds to a box filter (or average downsampling
    -- this is the same as AvgPool in Pytorch), order=1 to a triangle filter
    (or linear downsampling), order=2 to cubic downsampling, and so on.
    See https://richzhang.github.io/antialiased-cnns/ for more details.
    """

    def __init__(self, channels=None, factor=2, order=1):
        super(Downsample, self).__init__()
        assert factor > 1, "Downsampling factor must be > 1"
        self.stride = factor
        self.channels = channels
        self.order = order

        # Figure out padding and check params make sense
        # The padding is given by order*(factor-1)/2
        # so order*(factor-1) must be divisible by 2
        total_padding = order * (factor - 1)
        assert total_padding % 2 == 0, (
            "Misspecified downsampling parameters."
            "Downsampling factor and order must be such "
            "that order*(factor-1) is divisible by 2"
        )
        self.padding = int(order * (factor - 1) / 2)

        box_kernel = np.ones(factor)
        kernel = np.ones(factor)
        for _ in range(order):
            kernel = np.convolve(kernel, box_kernel)
        kernel /= np.sum(kernel)
        kernel = torch.Tensor(kernel)
        self.register_buffer(
            "kernel", kernel[None, None, :].repeat((channels, 1, 1))
        )

    def forward(self, x):
        return F.conv1d(
            x,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.shape[1],
        )

class ResBlock(nn.Module):
    r""" Basic bulding block in Resnets:

       bn-relu-conv-bn-relu-conv
      /                         \
    x --------------------------(+)->

    """

    def __init__(
        self, in_channels, out_channels, kernel_size=5, stride=1, padding=2
    ):

        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(x))
        x = self.conv1(x)
        x = self.relu(self.bn2(x))
        x = self.conv2(x)

        x = x + identity

        return x

class Resnet(nn.Module):
    r"""The general form of the architecture can be described as follows:

    x->[Conv-[ResBlock]^m-BN-ReLU-Down]^n->y

    In other words:

            bn-relu-conv-bn-relu-conv                        bn-
           /                         \                      /
    x->conv --------------------------(+)-bn-relu-down-> conv ----

    """

    def __init__(
        self,
        output_size=1,
        n_channels=3,
        resnet_version=1,
        epoch_len=10,
        add_classical_feats=False,
        num_classical_feats=22,
        sample_rate=30,
        my_device=None
    ):
        super(Resnet, self).__init__()

        # Architecture definition. Each tuple defines
        # a basic Resnet layer Conv-[ResBlock]^m]-BN-ReLU-Down
        # isEva: change the classifier to two FC with ReLu
        # For example, (64, 5, 1, 5, 3, 1) means:
        # - 64 convolution filters
        # - kernel size of 5
        # - 1 residual block (ResBlock)
        # - ResBlock's kernel size of 5
        # - downsampling factor of 3
        # - downsampling filter order of 1
        # In the below, note that 3*3*5*5*4 = 900 (input size)
        if resnet_version == 1:
            if epoch_len == 5:
                cgf = [
                    (64, 5, 2, 5, 2, 2),
                    (128, 5, 2, 5, 2, 2),
                    (256, 5, 2, 5, 3, 1),
                    (256, 5, 2, 5, 3, 1),
                    (512, 5, 0, 5, 3, 1),
                ]
            elif epoch_len == 10:
                cgf = [
                    (64, 5, 2, 5, 2, 2),
                    (128, 5, 2, 5, 2, 2),
                    (256, 5, 2, 5, 5, 1),
                    (512, 5, 2, 5, 5, 1),
                    (1024, 5, 0, 5, 3, 1),
                ]
            else:
                cgf = [
                    (64, 5, 2, 5, 3, 1),
                    (128, 5, 2, 5, 3, 1),
                    (256, 5, 2, 5, 5, 1),
                    (512, 5, 2, 5, 5, 1),
                    (1024, 5, 0, 5, 4, 0),
                ]
        else:
            cgf = [
                (64, 5, 2, 5, 3, 1),
                (64, 5, 2, 5, 3, 1),
                (128, 5, 2, 5, 5, 1),
                (128, 5, 2, 5, 5, 1),
                (256, 5, 2, 5, 4, 0),
            ]  # smaller resnet
        in_channels = n_channels
        feature_extractor = nn.Sequential()
        for i, layer_params in enumerate(cgf):
            (
                out_channels,
                conv_kernel_size,
                n_resblocks,
                resblock_kernel_size,
                downfactor,
                downorder,
            ) = layer_params
            feature_extractor.add_module(
                f"layer{i+1}",
                Resnet.make_layer(
                    in_channels,
                    out_channels,
                    conv_kernel_size,
                    n_resblocks,
                    resblock_kernel_size,
                    downfactor,
                    downorder,
                ),
            )
            in_channels = out_channels

        self.feature_extractor = feature_extractor
        self.add_classical_feats = add_classical_feats
        self.sample_rate = sample_rate
        self.my_device = my_device
        # Classifier input size = last out_channels in previous layer
        if self.add_classical_feats:
            out_channels = out_channels + num_classical_feats
        self.classifier = EvaClassifier(
            input_size=out_channels, output_size=output_size
        )
            # TODO: For Experiment 5
            # Approaches...
            # EvaClassifier
            # input_size = 1024 + # of handcraft_features
            # 
            # if cfg.model.varA or varB
            #   self.classifier = var_EvaClassifier()
        '''
        ...NOT USED...

        elif is_mtl:
            self.aot_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
            self.scale_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
            self.permute_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
            self.time_w_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
        else:
            self.classifier = Classifier(
                input_size=out_channels, output_size=output_size
            )
        '''

        weight_init(self)

    @staticmethod
    def make_layer(
        in_channels,
        out_channels,
        conv_kernel_size,
        n_resblocks,
        resblock_kernel_size,
        downfactor,
        downorder=1,
    ):
        r""" Basic layer in Resnets:

        x->[Conv-[ResBlock]^m-BN-ReLU-Down]->

        In other words:

                bn-relu-conv-bn-relu-conv
               /                         \
        x->conv --------------------------(+)-bn-relu-down->

        """

        # Check kernel sizes make sense (only odd numbers are supported)
        assert (
            conv_kernel_size % 2
        ), "Only odd number for conv_kernel_size supported"
        assert (
            resblock_kernel_size % 2
        ), "Only odd number for resblock_kernel_size supported"

        # Figure out correct paddings
        conv_padding = int((conv_kernel_size - 1) / 2)
        resblock_padding = int((resblock_kernel_size - 1) / 2)

        modules = [
            nn.Conv1d(
                in_channels,
                out_channels,
                conv_kernel_size,
                1,
                conv_padding,
                bias=False,
                padding_mode="circular",
            )
        ]

        for i in range(n_resblocks):
            modules.append(
                ResBlock(
                    out_channels,
                    out_channels,
                    resblock_kernel_size,
                    1,
                    resblock_padding,
                )
            )

        modules.append(nn.BatchNorm1d(out_channels))
        modules.append(nn.ReLU(True))
        # Dropout
        modules.append(nn.Dropout(p=0.5))
        modules.append(Downsample(out_channels, downfactor, downorder))

        return nn.Sequential(*modules)

    def forward(self, x):

        feats = self.feature_extractor(x)
        feats_flattened = feats.view(feats.shape[0], -1)

        if self.add_classical_feats:

            class_feats = handcraft_features_batch(x, self.sample_rate).to(self.my_device, dtype=torch.float)
            class_feats_flattened = class_feats.view(class_feats.shape[0], -1)

            # Normalize ResNet features column-wise (batch size, 1024)
            feats_flattened = F.normalize(feats_flattened,dim=0)
            # Normalize Classical features column-wise (batch size, 22)
            class_feats_flattened = F.normalize(class_feats_flattened,dim=0)

            final_feats = torch.cat((feats_flattened, class_feats_flattened),1)

        else:
            final_feats = feats_flattened
        
        y = self.classifier(final_feats)

        return y, final_feats

def weight_init(self, mode="fan_out", nonlinearity="relu"):

    for m in self.modules():

        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(
                m.weight, mode=mode, nonlinearity=nonlinearity
            )

        elif isinstance(m, (nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


# TODO: For Experiment 5
# Approaches...
# ...

def handcraft_features_batch(xyz_batch:torch.Tensor, sample_rate:int) -> torch.Tensor:

    feats_batch = []

    for xyz in xyz_batch:
        xyz_array = xyz.cpu().numpy().T # Sync the data shape from (3,N) to (N,3)
        feats = handcraft_features(xyz_array, sample_rate)
        feats_batch.append(list(feats.values()))

    # print(f"*** classical features of Last batch window\n{feats}")

    return torch.Tensor(feats_batch)

def handcraft_features(xyz:np.ndarray, sample_rate:int) -> dict:

    """Our baseline handcrafted features. xyz is a window of shape (N,3)"""

    feats = {}
    feats["xMean"], feats["yMean"], feats["zMean"] = np.mean(xyz, axis=0)
    feats["xStd"], feats["yStd"], feats["zStd"] = np.std(xyz, axis=0)
    feats["xRange"], feats["yRange"], feats["zRange"] = np.ptp(xyz, axis=0)

    x, y, z = xyz.T

    with np.errstate(
        divide="ignore", invalid="ignore"
    ):  # ignore div by 0 warnings
        feats["xyCorr"] = np.nan_to_num(np.corrcoef(x, y)[0, 1])
        feats["yzCorr"] = np.nan_to_num(np.corrcoef(y, z)[0, 1])
        feats["zxCorr"] = np.nan_to_num(np.corrcoef(z, x)[0, 1])

    m = np.linalg.norm(xyz, axis=1)

    feats["mean"] = np.mean(m)
    feats["std"] = np.std(m)
    feats["range"] = np.ptp(m)
    feats["mad"] = stats.median_abs_deviation(m)
    feats["kurt"] = stats.kurtosis(m)
    feats["skew"] = stats.skew(m)
    feats["enmomean"] = np.mean(np.abs(m - 1))

    # Spectrum using Welch's method with 3s segment length
    # First run without detrending to get the true spectrum
    freqs, powers = signal.welch(
        m,
        fs=sample_rate,
        nperseg=3 * sample_rate,
        noverlap=2 * sample_rate,
        detrend=False,
        average="median",
    )

    with np.errstate(
        divide="ignore", invalid="ignore"
    ):  # ignore div by 0 warnings
        feats["pentropy"] = np.nan_to_num(stats.entropy(powers + 1e-16))

    # Spectrum using Welch's method with 3s segment length
    # Now do detrend to find dominant freqs
    freqs, powers = signal.welch(
        m,
        fs=sample_rate,
        nperseg=3 * sample_rate,
        noverlap=2 * sample_rate,
        detrend="constant",
        average="median",
    )

    peaks, _ = signal.find_peaks(powers)
    peak_powers = powers[peaks]
    peak_freqs = freqs[peaks]
    peak_ranks = np.argsort(peak_powers)[::-1]
    if len(peaks) >= 2:
        feats["f1"] = peak_freqs[peak_ranks[0]]
        feats["f2"] = peak_freqs[peak_ranks[1]]
    elif len(peaks) == 1:
        feats["f1"] = feats["f2"] = peak_freqs[peak_ranks[0]]
    else:
        feats["f1"] = feats["f2"] = 0

    return feats