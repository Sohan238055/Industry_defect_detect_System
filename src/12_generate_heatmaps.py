
from pathlib import Path
import gc

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

# ==========================================
# CONFIGURATION
# ==========================================

DATASET_DIR = Path("dataset")
CHECKPOINT_DIR = Path("results/Patchcore_Fast")
OUTPUT_DIR = Path("results/heatmaps")

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# DEVICE
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ==========================================
# PROCESS EACH CATEGORY
# ==========================================

completed = 0

for category in CATEGORIES:

    print("\nProcessing:", category)

    try:
        # Find existing checkpoint
        checkpoints = list(
            (CHECKPOINT_DIR / category).rglob("model.ckpt")
        )

        if not checkpoints:
            print("Checkpoint not found. Skipping:", category)
            continue

        checkpoint = checkpoints[0]

        # Find one defective test image
        test_dir = DATASET_DIR / category / "test"

        defect_images = []

        for folder in test_dir.iterdir():
            if folder.is_dir() and folder.name.lower() != "good":
                defect_images.extend(
                    list(folder.glob("*.png"))
                )
                defect_images.extend(
                    list(folder.glob("*.jpg"))
                )

        if not defect_images:
            print("No defective test image found. Skipping.")
            continue

        image_path = sorted(defect_images)[0]

        print("Image:", image_path.name)
        print("Checkpoint:", checkpoint)

        # Prepare prediction dataset
        dataset = PredictDataset(
            path=str(image_path),
            image_size=(256, 256)
        )

        # Load existing PatchCore model
        model = Patchcore()

        # Avoid automatic visualization output
        model.visualizer = None

        engine = Engine(
            accelerator=device,
            devices=1
        )

        predictions = engine.predict(
            model=model,
            dataset=dataset,
            ckpt_path=str(checkpoint),
            return_predictions=True
        )

        if not predictions:
            print("No predictions returned.")
            continue

        prediction = predictions[0]

        # Extract anomaly map
        anomaly_map = prediction.anomaly_map

        if isinstance(anomaly_map, torch.Tensor):
            anomaly_map = anomaly_map.detach().cpu().numpy()

        anomaly_map = np.squeeze(anomaly_map)

        # Load original image
        original = Image.open(image_path).convert("RGB")
        width, height = original.size

        # Resize anomaly map to original image dimensions
        heatmap = Image.fromarray(
            anomaly_map.astype(np.float32)
        )

        heatmap = heatmap.resize(
            (width, height),
            Image.Resampling.BILINEAR
        )

        heatmap = np.array(heatmap)

        # Generate overlay
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(original)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(original)
        plt.imshow(heatmap, cmap="jet", alpha=0.5)
        plt.title("PatchCore Fast Defect Heatmap")
        plt.axis("off")

        plt.tight_layout()

        output_path = OUTPUT_DIR / f"{category}_heatmap.png"

        plt.savefig(
            output_path,
            dpi=100,
            bbox_inches="tight"
        )

        plt.close()

        print("Saved:", output_path)
        completed += 1

        # Release memory
        del predictions, prediction, dataset, engine, model
        del anomaly_map, heatmap, original
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as error:
        print("ERROR in", category, ":", error)
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print("\n" + "=" * 50)
print("HEATMAP GENERATION FINISHED")
print("Completed:", completed, "/", len(CATEGORIES))
print("Output:", OUTPUT_DIR)
print("=" * 50)