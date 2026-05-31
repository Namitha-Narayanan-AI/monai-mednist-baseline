import matplotlib.pyplot as plt
import torch

from src.config import FIGURES_DIR
from src.data import create_dataloaders


def show_sample_batch(num_images: int = 12) -> None:
    """
    Saves a grid of sample MedNIST images with their class labels.
    This is used to visually verify that the dataset and labels are loaded correctly.
    """
    train_loader, _, _, class_names = create_dataloaders()

    images, labels = next(iter(train_loader))

    num_images = min(num_images, len(images))

    cols = 4
    rows = (num_images + cols - 1) // cols

    plt.figure(figsize=(12, 3 * rows))

    for index in range(num_images):
        image = images[index].squeeze(0)
        label_index = labels[index].item()
        label_name = class_names[label_index]

        plt.subplot(rows, cols, index + 1)
        plt.imshow(image, cmap="gray")
        plt.title(label_name)
        plt.axis("off")

    plt.tight_layout()

    output_path = FIGURES_DIR / "sample_mednist_images.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved sample image grid to: {output_path}")


if __name__ == "__main__":
    show_sample_batch()