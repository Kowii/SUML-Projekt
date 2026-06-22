"""Two-phase fine-tuning of a ResNet backbone on the CUB-200-2011 bird dataset."""

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

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

_PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_polish_names(classes: list[str]) -> list[str]:
    """Return Polish display names for each class folder name, falling back to the folder name."""
    default_names_path = str(_PROJECT_ROOT / "data" / "bird_names_pl.json")
    names_path = Path(os.getenv("BIRD_NAMES_FILE", default_names_path))
    if not names_path.exists():
        logger.warning("Polish names file not found at %s — using raw folder names.", names_path)
        return classes

    with open(names_path, encoding="utf-8") as f:
        mapping: dict[str, str] = json.load(f)

    display_names = [mapping.get(cls, cls) for cls in classes]
    missing = [cls for cls in classes if cls not in mapping]
    if missing:
        logger.warning("%d classes had no Polish translation: %s", len(missing), missing[:5])
    return display_names


def build_model(model_name: str, num_classes: int) -> nn.Module:
    """Return a pretrained backbone with frozen feature extractor and a new classification head."""
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(_MODEL_REGISTRY)}")

    model_fn, weights, fc_in_features = _MODEL_REGISTRY[model_name]
    model = model_fn(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(fc_in_features, num_classes)
    return model


def run_epoch(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str = "",
) -> tuple[float, float]:
    """Run one training or validation epoch; return (avg_loss, top1_accuracy)."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_training):
        for images, labels in tqdm(loader, desc=desc, leave=False, ncols=90):
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


def _log_epoch(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    phase_label: str,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    best_val: float,
    elapsed: float,
) -> None:
    """Print a single formatted epoch summary line."""
    logger.info(
        "[%s | Epoch %d/%d] "
        "train_loss=%.4f train_acc=%.1f%% | "
        "val_loss=%.4f val_acc=%.1f%% | "
        "best_val=%.1f%% | time=%.0fs",
        phase_label, epoch, total_epochs,
        train_loss, train_acc * 100,
        val_loss, val_acc * 100,
        best_val * 100,
        elapsed,
    )


def _generate_report(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    output_path: str,
    model_name: str,
    device: torch.device,
    batch_size: int,
    phase1_epochs: int,
    phase2_epochs: int,
    phase1_lr: float,
    phase2_lr: float,
    num_train: int,
    num_val: int,
    num_classes: int,
    history: list[dict],
    best_val_acc: float,
    best_epoch: int,
    total_seconds: float,
) -> None:
    """Write a Polish-language Markdown training report to output_path."""
    hours, remainder = divmod(int(total_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_str = f"{hours}h {minutes}m {seconds}s" if hours else f"{minutes}m {seconds}s"

    lines = [
        "# Raport trenowania modelu OrnithoAI",
        "",
        "## Informacje ogólne",
        "",
        f"- **Model:** {model_name}",
        f"- **Urządzenie:** {device}",
        f"- **Liczba klas:** {num_classes}",
        f"- **Zbiór treningowy:** {num_train} próbek",
        f"- **Zbiór walidacyjny:** {num_val} próbek",
        f"- **Czas trenowania:** {duration_str}",
        "",
        "## Hiperparametry",
        "",
        f"- **Rozmiar batcha:** {batch_size}",
        f"- **Faza 1 (tylko głowica):** {phase1_epochs} epok, lr={phase1_lr}",
        f"- **Faza 2 (pełny fine-tuning):** {phase2_epochs} epok, lr={phase2_lr}",
        "- **Scheduler:** CosineAnnealingLR (faza 2)",
        "- **Funkcja kosztu:** CrossEntropyLoss z label smoothing=0.1",
        "",
        "## Wyniki per-epoka",
        "",
        "| Faza | Epoka | Train Acc | Val Acc | Train Loss | Val Loss |",
        "|------|-------|-----------|---------|------------|----------|",
    ]

    for row in history:
        lines.append(
            f"| {row['phase']} | {row['epoch']:>5} "
            f"| {row['train_acc'] * 100:>8.1f}% "
            f"| {row['val_acc'] * 100:>6.1f}% "
            f"| {row['train_loss']:>10.4f} "
            f"| {row['val_loss']:>8.4f} |"
        )

    lines += [
        "",
        "## Podsumowanie",
        "",
        f"- **Najlepsza dokładność walidacyjna:** {best_val_acc * 100:.2f}%",
        f"- **Najlepsza epoka:** {best_epoch}",
        "",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Raport trenowania zapisany: %s", output_path)


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Run two-phase fine-tuning: frozen head training, then full backbone fine-tuning."""
    skip_train = os.getenv("SKIP_TRAIN", "false").lower() in ("true", "1", "yes")
    if skip_train:
        logger.info("SKIP_TRAIN is set to true. Skipping training execution.")
        return

    dataset_dir = os.getenv("DATASET_DIR", str(_PROJECT_ROOT / "dataset" / "CUB_200_2011" / "images"))
    model_dir = os.getenv("MODEL_DIR", str(_PROJECT_ROOT / "models"))
    model_file = os.getenv("MODEL_FILE", "bird_classifier.pt")
    classes_file = os.getenv("CLASSES_FILE", "classes.json")
    report_file = os.getenv("REPORT_FILE", "training_report.md")
    model_name = os.getenv("MODEL_NAME", "resnet50")
    batch_size = int(os.getenv("BATCH_SIZE", "64"))
    phase1_epochs = int(os.getenv("PHASE1_EPOCHS", "5"))
    phase2_epochs = int(os.getenv("PHASE2_EPOCHS", "20"))
    phase1_lr = float(os.getenv("PHASE1_LR", "1e-3"))
    phase2_lr = float(os.getenv("PHASE2_LR", "1e-4"))

    model_path = os.path.join(model_dir, model_file)
    classes_path = os.path.join(model_dir, classes_file)
    report_path = os.path.join(model_dir, report_file)
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("GPU: %s (%.1f GB VRAM)", gpu_name, vram_gb)
    else:
        logger.info("Brak GPU — trenowanie na CPU (będzie wolno).")

    logger.info("Ładowanie datasetu z: %s", dataset_dir)
    train_dataset, val_dataset, classes = load_datasets(dataset_dir)
    logger.info(
        "Klasy: %d | Trening: %d próbek | Walidacja: %d próbek",
        len(classes), len(train_dataset), len(val_dataset),
    )

    display_names = load_polish_names(classes)

    num_workers = min(4, os.cpu_count() or 1)
    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
    )

    model = build_model(model_name, num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history: list[dict] = []
    best_val_acc = 0.0
    best_epoch = 0
    training_start = time.time()

    # ── Faza 1: tylko głowica ────────────────────────────────────────────────
    logger.info("=== Faza 1: trenowanie głowicy (%d epok, lr=%.4f) ===", phase1_epochs, phase1_lr)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=phase1_lr)

    for epoch in range(1, phase1_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device,
            desc=f"P1 E{epoch}/{phase1_epochs} train",
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device,
            desc=f"P1 E{epoch}/{phase1_epochs} val",
        )
        elapsed = time.time() - t0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
            logger.info("  -> Nowy najlepszy checkpoint (val_acc=%.1f%%)", val_acc * 100)

        _log_epoch("Faza 1", epoch, phase1_epochs, train_loss, train_acc,
                   val_loss, val_acc, best_val_acc, elapsed)
        history.append({"phase": "1", "epoch": epoch, "train_loss": train_loss,
                        "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

    # ── Faza 2: pełny fine-tuning ────────────────────────────────────────────
    logger.info("=== Faza 2: pełny fine-tuning (%d epok, lr=%.5f) ===", phase2_epochs, phase2_lr)
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=phase2_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_epochs)

    for epoch in range(1, phase2_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device,
            desc=f"P2 E{epoch}/{phase2_epochs} train",
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device,
            desc=f"P2 E{epoch}/{phase2_epochs} val",
        )
        scheduler.step()
        elapsed = time.time() - t0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = phase1_epochs + epoch
            torch.save(model.state_dict(), model_path)
            logger.info("  -> Nowy najlepszy checkpoint (val_acc=%.1f%%)", val_acc * 100)

        _log_epoch("Faza 2", epoch, phase2_epochs, train_loss, train_acc,
                   val_loss, val_acc, best_val_acc, elapsed)
        history.append({"phase": "2", "epoch": epoch, "train_loss": train_loss,
                        "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

    # ── Zapis klas i raportu ────────────────────────────────────────────────
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(display_names, f, ensure_ascii=False, indent=2)

    total_seconds = time.time() - training_start
    _generate_report(
        output_path=report_path,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        phase1_epochs=phase1_epochs,
        phase2_epochs=phase2_epochs,
        phase1_lr=phase1_lr,
        phase2_lr=phase2_lr,
        num_train=len(train_dataset),
        num_val=len(val_dataset),
        num_classes=len(classes),
        history=history,
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        total_seconds=total_seconds,
    )

    logger.info("Trenowanie zakończone. Najlepsza val_acc: %.1f%% (epoka %d)",
                best_val_acc * 100, best_epoch)
    logger.info("Model zapisany: %s", model_path)
    logger.info("Klasy zapisane: %s", classes_path)


if __name__ == "__main__":
    main()
