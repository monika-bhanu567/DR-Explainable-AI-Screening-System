from pathlib import Path
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


class IDRiDDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = Path(image_dir)
        self.labels = pd.read_csv(label_file)
        self.transform = transform

        # Keep only the columns we need
        self.labels = self.labels[["Image name", "Retinopathy grade"]]

        # Remove any missing values
        self.labels = self.labels.dropna()

        # Convert labels to integers
        self.labels["Retinopathy grade"] = (
            self.labels["Retinopathy grade"].astype(int)
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        row = self.labels.iloc[index]

        image_name = row["Image name"]
        label = int(row["Retinopathy grade"])

        # Find image
        image_path = self.image_dir / f"{image_name}.jpg"

        # Read image
        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        # OpenCV uses BGR, Albumentations expects RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentation / preprocessing
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Convert image to PyTorch tensor
        image = torch.tensor(
            image,
            dtype=torch.float32
        ).permute(2, 0, 1) / 255.0

        return image, torch.tensor(label, dtype=torch.long)


def create_dataloaders(
    train_image_dir,
    train_label_file,
    validation_image_dir,
    validation_label_file,
    train_transform,
    validation_transform,
    batch_size=16
):

    train_dataset = IDRiDDataset(
        image_dir=train_image_dir,
        label_file=train_label_file,
        transform=train_transform
    )

    validation_dataset = IDRiDDataset(
        image_dir=validation_image_dir,
        label_file=validation_label_file,
        transform=validation_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, validation_loader