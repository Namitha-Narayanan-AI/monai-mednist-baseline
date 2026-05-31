import json

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

from src.config import DEVICE, FIGURES_DIR, METRICS_DIR, MODELS_DIR
from src.data import create_dataloaders
from src.model import MedNISTCNN


def evaluate_model(model, dataloader):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_predictions


def main():
    print(f"Using device: {DEVICE}")

    _, _, test_loader, class_names = create_dataloaders()

    model_path = MODELS_DIR / "best_mednist_cnn.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)

    model = MedNISTCNN(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    true_labels, predicted_labels = evaluate_model(
        model=model,
        dataloader=test_loader,
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        output_dict=True,
    )

    report_path = METRICS_DIR / "classification_report.json"

    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    print("\nClassification Report:")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=class_names,
        )
    )

    cm = confusion_matrix(true_labels, predicted_labels)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    display.plot(xticks_rotation=45)
    plt.tight_layout()

    confusion_matrix_path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(confusion_matrix_path, dpi=300)
    plt.close()

    print(f"Saved classification report to: {report_path}")
    print(f"Saved confusion matrix to: {confusion_matrix_path}")


if __name__ == "__main__":
    main()