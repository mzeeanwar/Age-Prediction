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

def main():
    args = parse_args()
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    dataset = UTKFaceDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = AgeEstimator(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()  # MAE in years, easy to interpret

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, ages in tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            images, ages = images.to(device), ages.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, ages)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        print(f"Epoch {epoch + 1}: MAE = {total_loss / len(dataset):.2f} years")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
