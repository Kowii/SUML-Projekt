"""Fine-tune a frozen ResNet backbone on the local bird image dataset."""

import json
import logging
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

# data/ module lives at the project root; resolved via PYTHONPATH at runtime
from data.dataset import load_datasets  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("trainer")

_MODEL_REGISTRY: dict[str, tuple] = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1, 2048),
}


def build_model(model_name: str, num_classes: int) -> nn.Module:
    """Return a pretrained backbone with a frozen feature extractor and a new classification head."""
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(_MODEL_REGISTRY)}")

    model_fn, weights, fc_in_features = _MODEL_REGISTRY[model_name]
    model = model_fn(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(fc_in_features, num_classes)
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training or validation epoch and return (avg_loss, top1_accuracy)."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += len(labels)

    return total_loss / total, correct / total


def main() -> None:
    """Load bird images, fine-tune backbone head for N epochs, save the best model checkpoint."""
    dataset_dir = os.getenv("DATASET_DIR", "/dataset")
    model_dir = os.getenv("MODEL_DIR", "/models")
    model_file = os.getenv("MODEL_FILE", "bird_classifier.pt")
    classes_file = os.getenv("CLASSES_FILE", "classes.json")
    model_name = os.getenv("MODEL_NAME", "resnet18")
    batch_size = int(os.getenv("BATCH_SIZE", "32"))
    num_epochs = int(os.getenv("NUM_EPOCHS", "5"))
    learning_rate = float(os.getenv("LEARNING_RATE", "1e-3"))

    model_path = os.path.join(model_dir, model_file)
    classes_path = os.path.join(model_dir, classes_file)
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    logger.info("Loading dataset from: %s", dataset_dir)
    train_dataset, val_dataset, classes = load_datasets(dataset_dir)
    logger.info(
        "Classes: %s | Train: %d samples | Val: %d samples",
        classes, len(train_dataset), len(val_dataset),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_model(model_name, num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

    logger.info("Starting training: %s, %d epochs, lr=%.4f", model_name, num_epochs, learning_rate)
    best_val_accuracy = 0.0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)

        logger.info(
            "Epoch %d/%d | Train loss: %.4f acc: %.1f%% | Val loss: %.4f acc: %.1f%%",
            epoch, num_epochs,
            train_loss, train_acc * 100,
            val_loss, val_acc * 100,
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), model_path)
            logger.info("  -> New best checkpoint saved (val acc: %.1f%%)", val_acc * 100)

    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    logger.info("Training complete. Best val accuracy: %.1f%%", best_val_accuracy * 100)
    logger.info("Model saved to: %s", model_path)
    logger.info("Classes saved to: %s", classes_path)


if __name__ == "__main__":
    main()