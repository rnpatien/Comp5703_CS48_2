import csv
import pathlib

# from typing import Any, Callable, Optional, Tuple
# import PIL
# from .folder import make_dataset
# from .utils import download_and_extract_archive, verify_str_arg
# from .vision import VisionDataset
import torch
from torchvision import  transforms
from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np


class CustomCIFAR10(CIFAR10):
    def __init__(self, sublist, **kwds):
        super().__init__(**kwds)

        if len(sublist) > 0:
            self.data = self.data[sublist]
            self.targets = np.array(self.targets)[sublist].tolist()

        self.idxsPerClass = [np.where(np.array(self.targets) == idx)[0] for idx in range(10)]
        self.idxsNumPerClass = [len(idxs) for idxs in self.idxsPerClass]
        return

    # def __getitem__(self, idx):
    #     img = self.data[idx]
    #     img = Image.fromarray(img).convert('RGB')
    #     imgs = [self.transform(img), self.transform(img)]

    #     return torch.stack(imgs)
