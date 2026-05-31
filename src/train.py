import json

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from src.config import (
    DEVICE,
    LEARNING_RATE,
    METRICS_DIR,
    MODELS_DIR,
    NUM_EPOCHS,
)
from src.data import create_dataloaders
from src.model import MedNISTCNN
from src.utils import set_seed


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        predictions = torch.argmax(outputs, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


def validate(model, dataloader, criterion):
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


def main():
    set_seed()

    print(f"Using device: {DEVICE}")

    train_loader, val_loader, _, class_names = create_dataloaders()

    model = MedNISTCNN(num_classes=len(class_names)).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = 0.0
    history = []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )

        val_loss, val_accuracy = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
        )

        epoch_result = {
            "epoch": int(epoch + 1),
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_accuracy),
        }

        history.append(epoch_result)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = float(val_accuracy)

            model_path = MODELS_DIR / "best_mednist_cnn.pth"

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "val_accuracy": best_val_accuracy,
                },
                model_path,
            )

            print(f"Saved best model to: {model_path}")

    history_path = METRICS_DIR / "training_history.json"

    with open(history_path, "w") as file:
        json.dump(history, file, indent=4)

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Saved training history to: {history_path}")


if __name__ == "__main__":
    main()