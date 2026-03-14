import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class WFLWDataset(Dataset):
    def __init__(self, img_dir, anno_file, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.annos = self._read_txt_file(anno_file)

    def __len__(self):
        return len(self.annos)
    
    def __getitem__(self, idx):
        anno = self.annos[idx]
        img_path = self.img_dir / anno["img_name"]

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        landmarks = anno["landmarks"]

        if self.transform:
            transformed = self.transform(image=img_np, keypoints=landmarks)
            img_tensor = transformed["image"]
            landmarks = transformed["keypoints"]
            landmarks_np = np.array(landmarks, dtype=np.float32)
        else:
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        landmarks_np = landmarks_np / 255.0
        landmarks_tensor = torch.tensor(landmarks_np, dtype=torch.float32).view(-1)

        return img_tensor, landmarks_tensor

    def _read_txt_file(self, txt_file):
        data = []
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                landmarks = np.array(parts[:196], dtype=np.float32).reshape(98, 2)
                img_name = parts[-1]
                data.append({
                    "img_name": img_name,
                    "landmarks": landmarks
                })
        
        return data