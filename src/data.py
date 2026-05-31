from pathlib import Path
from typing import List, Tuple

import torch
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensity,
    Resize,
    ToTensor,
)
from torch.utils.data import Dataset, DataLoader, random_split

from src.config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    MEDNIST_DIR,
    RANDOM_SEED,
)


class MedNISTDataset(Dataset):
    """
    Custom dataset for MedNIST.

    Each sample contains:
    - image path
    - numeric class label
    """

    def __init__(self, image_paths: List[Path], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = str(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms() -> Compose:
    """
    Defines preprocessing steps applied to each image before it enters the model.
    """
    return Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensity(),
            Resize((IMAGE_SIZE, IMAGE_SIZE)),
            ToTensor(),
        ]
    )


def get_image_paths_and_labels() -> Tuple[List[Path], List[int], List[str]]:
    """
    Reads the MedNIST folder structure and creates image paths and numeric labels.

    Folder name becomes the class name.
    Example:
    AbdomenCT -> 0
    BreastMRI -> 1
    CXR -> 2
    """
    class_names = sorted(
        [
            folder.name
            for folder in MEDNIST_DIR.iterdir()
            if folder.is_dir()
        ]
    )

    image_paths = []
    labels = []

    for label_index, class_name in enumerate(class_names):
        class_dir = MEDNIST_DIR / class_name

        class_images = (
            list(class_dir.glob("*.jpeg"))
            + list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.png"))
        )

        for image_path in class_images:
            image_paths.append(image_path)
            labels.append(label_index)

    return image_paths, labels, class_names


def get_dataset() -> Tuple[MedNISTDataset, List[str]]:
    """
    Creates the full MedNIST dataset.
    """
    image_paths, labels, class_names = get_image_paths_and_labels()

    dataset = MedNISTDataset(
        image_paths=image_paths,
        labels=labels,
        transform=get_transforms(),
    )

    return dataset, class_names


def create_dataloaders(
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Splits the dataset into training, validation, and test sets.
    Returns dataloaders and class names.
    """
    dataset, class_names = get_dataset()

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = create_dataloaders()

    print("Classes:", class_names)
    print("Number of classes:", len(class_names))
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    images, labels = next(iter(train_loader))

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Sample labels:", labels[:10])
    print("Image dtype:", images.dtype)
    print("Image min:", images.min().item())
    print("Image max:", images.max().item())