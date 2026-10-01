"""Train the age-regression CNN on UTKFace-style data.

Expects a directory of images named like: <age>_<gender>_<race>_<date>.jpg
"""

import argparse
import glob
import os

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from age_estimator import AgeEstimator, PREPROCESS


class UTKFaceDataset(Dataset):
    def __init__(self, data_dir):
        self.paths = glob.glob(os.path.join(data_dir, "*.jpg"))
        if not self.paths:
            raise RuntimeError(f"No .jpg files found in {data_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        age = int(os.path.basename(path).split("_")[0])
        image = cv2.imread(path)
        image = PREPROCESS(image)
        return image, torch.tensor(age, dtype=torch.float32)
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", default="models/age_model.pt")
    p.add_argument("--cuda", action="store_true")
    return p.parse_args()
