# Source from: https://github.com/OxWearables/ssl-wearables
# modifided by Hyeongkyun Kim(hyeongkyun.kim@uzh.ch) and Orestis Oikonomou(orestis.oikonomou@uzh.ch)
# ---------------------------------------------------------------
# Copyright © 2022, University of Oxford
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
# A copy of the license is available at resources/license_mtssl

import torch

class NormalDataset:
    def __init__(
        self,
        X,
        y=[],
        pid=[],
        name="",
        isLabel=False,
        transform=None,
        target_transform=None,
    ):
        """
        Y needs to be in one-hot encoding
        X needs to be in N * Width
        Args:
            data_path (string): path to data
            files_to_load (list): subject names
            currently all npz format should allow support multiple ext

        """

        self.X = torch.from_numpy(X)
        self.y = y
        self.isLabel = isLabel
        self.transform = transform
        self.targetTransform = target_transform
        self.pid = pid
        print(name + " set sample count : " + str(len(self.X)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.X[idx, :]
        y = []
        if self.isLabel:
            y = self.y[idx]
            if self.targetTransform:
                y = self.targetTransform(y)

        if self.transform:
            sample = self.transform(sample)
        if len(self.pid) >= 1:
            return sample, y, self.pid[idx]
        else:
            return sample, y
