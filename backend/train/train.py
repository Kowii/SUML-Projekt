import json
import logging
import os
import sys

import timm
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("trainer")

def main():
    logger.info("Initializing Model Generator (Training Skeleton)...")

    # Load config from environment variables
    model_dir = os.getenv("MODEL_DIR", "/models")
    model_file = os.getenv("MODEL_FILE", "bird_classifier.pt")
    classes_file = os.getenv("CLASSES_FILE", "classes.json")
    model_name = os.getenv("MODEL_NAME", "resnet50")

    model_path = os.path.join(model_dir, model_file)
    classes_path = os.path.join(model_dir, classes_file)

    os.makedirs(model_dir, exist_ok=True)

    # 1. Define classes and save mapping
    classes = ["sparrow", "robin", "pigeon", "eagle", "seagull"]
    logger.info("Writing classes mapping to: %s", classes_path)
    with open(classes_path, "w") as f:
        json.dump(classes, f, indent=2)
    logger.info("Class mapping successfully saved.")

    # 2. Instantiate and save model weights
    logger.info("Instantiating pretrained model '%s' using timm...", model_name)
    try:
        # TIMM will fetch pretrained ImageNet weights, and automatically reset the
        # classification layer ('head') to length of classes (5).
        model = timm.create_model(model_name, pretrained=True, num_classes=len(classes))
        logger.info("Pretrained weights successfully downloaded and head initialized.")

        logger.info("Saving model state dictionary to: %s", model_path)
        torch.save(model.state_dict(), model_path)
        logger.info("Model weights successfully saved.")
        logger.info("Dummy pretrained model generation completed successfully!")
    except Exception as e:
        logger.error("Error occurred while generating dummy model: %s", str(e), exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
