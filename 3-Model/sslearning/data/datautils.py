# Source from: https://github.com/OxWearables/ssl-wearables
# modifided by Hyeongkyun Kim(hyeongkyun.kim@uzh.ch) and Orestis Oikonomou(orestis.oikonomou@uzh.ch)
# ---------------------------------------------------------------
# Copyright © 2022, University of Oxford
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
# A copy of the license is available at resources/license_mtssl

import torch
import numpy as np
from transforms3d.axangles import axangle2mat  # for rotation
import random


class RotationAxis(object):
    """
    Rotation along an axis
    """

    def __call__(self, sample):
        # 3 * FEATURE_SIZE
        sample = np.swapaxes(sample, 0, 1)
        angle = np.random.uniform(low=-np.pi, high=np.pi)
        axis = np.random.uniform(low=-1, high=1, size=sample.shape[1])
        sample = np.matmul(sample, axangle2mat(axis, angle))
        sample = np.swapaxes(sample, 0, 1)
        return sample

class RandomSwitchAxis(object):
    """
    Randomly switch the three axises for the raw files
    Input size: 3 * FEATURE_SIZE
    """

    def __call__(self, sample):
        # print(sample.shape)
        # 3 * FEATURE
        x = sample[0, :]
        y = sample[1, :]
        z = sample[2, :]

        choice = random.randint(1, 6)

        if choice == 1:
            sample = torch.stack([x, y, z], dim=0)
        elif choice == 2:
            sample = torch.stack([x, z, y], dim=0)
        elif choice == 3:
            sample = torch.stack([y, x, z], dim=0)
        elif choice == 4:
            sample = torch.stack([y, z, x], dim=0)
        elif choice == 5:
            sample = torch.stack([z, x, y], dim=0)
        elif choice == 6:
            sample = torch.stack([z, y, x], dim=0)
        # print(sample.shape)
        return sample

class AmplitudeScale(object):
    """
    Scaling up/down sample amplitude of all axis.
    """
    def __call__(self, sample):
        scaleFactor = np.random.uniform(low=0.6, high=1.4)
        sample = sample * scaleFactor
        return sample
