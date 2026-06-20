"""BirdDataset and helpers for loading a per-class image folder into PyTorch splits."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.2),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class BirdDataset(Dataset):
    """PyTorch Dataset for bird species images stored in per-class subdirectories."""

    def __init__(self, samples: list[tuple[Path, int]], transform: transforms.Compose):
        """Store pre-split sample list and the transform applied on each image load."""
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        """Return the total number of images in this dataset split."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Load the image at index, convert to RGB, apply transform, and return with label."""
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), label


def load_datasets(  # pylint: disable=too-many-locals
    data_dir: str | Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple["BirdDataset", "BirdDataset", list[str]]:
    """Scan class subdirectories, split samples, return (train_dataset, val_dataset, classes).

    Expects data_dir/{class_name}/*.jpg structure.
    """
    data_dir = Path(data_dir)
    class_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())

    if not class_dirs:
        raise FileNotFoundError(f"No class subdirectories found in '{data_dir}'.")

    classes = [d.name for d in class_dirs]
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    all_samples: list[tuple[Path, int]] = []
    for class_dir in class_dirs:
        label = class_to_idx[class_dir.name]
        images = [p for p in class_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        all_samples.extend((img, label) for img in images)

    if not all_samples:
        raise FileNotFoundError(
            f"No images found under '{data_dir}'. "
            f"Place images in subdirectories named after each class."
        )

    val_count = max(1, int(len(all_samples) * val_ratio))
    train_count = len(all_samples) - val_count

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        range(len(all_samples)), [train_count, val_count], generator=generator
    )

    train_samples = [all_samples[i] for i in train_subset]
    val_samples = [all_samples[i] for i in val_subset]

    return (
        BirdDataset(train_samples, TRAIN_TRANSFORMS),
        BirdDataset(val_samples, VAL_TRANSFORMS),
        classes,
    )
