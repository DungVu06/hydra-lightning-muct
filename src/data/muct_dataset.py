import os
import pandas as pd
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

class MUCTDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = str(self.data.iloc[idx, 1]) + ".jpg"
        img_path = self.img_dir / img_name
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        raw_landmarks = self.data.iloc[idx, 3:].values.astype("float32")
        landmarks = raw_landmarks.reshape(-1, 2)

        if self.transform:
            transformed = self.transform(image=img_np, keypoints=landmarks)
            img = transformed("image")
            landmarks = np.array(transformed["keypoints"]).flatten()

        return img, torch.tensor(landmarks, dtype=torch.float32)