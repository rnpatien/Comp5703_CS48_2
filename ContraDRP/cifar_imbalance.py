import csv
import pathlib

# from typing import Any, Callable, Optional, Tuple
# import PIL
# from .folder import make_dataset
# from .utils import download_and_extract_archive, verify_str_arg
# from .vision import VisionDataset
import torch
from torchvision import  transforms
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

def find_classes(directory: str) :
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

class cifarImbalanceataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,  root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self._samples = self.make_dataset(root_dir) 


    def make_dataset(self,dir):
        cls,class_to_idx=find_classes(dir)
        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(dir, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    samplenp = np.load(path)
                    item = samplenp, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)
        return instances

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        samplenp, target = self._samples[idx]
        # samplenp = np.load(path)
        # sample= np.resize(sample,(3,32,32))
        samplenp=np.transpose(np.reshape(samplenp,(3, 32,32)), (1,2,0))
        sample = self.transform(samplenp)
        return sample, target

