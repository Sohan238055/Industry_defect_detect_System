
from pathlib import Path
import csv
import time
import gc
import shutil
import tempfile
import torch

from anomalib.models import Patchcore, Dinomaly
from anomalib.data import PredictDataset
from anomalib.engine import Engine

DATASET = Path("dataset")
RESULTS = Path("results")
OUTPUT = RESULTS / "inference_benchmark.csv"

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

IMAGE_SIZE = (256, 256)
MAX_IMAGES = 5
WARMUP_IMAGES = 2
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

torch.set_float32_matmul_precision("medium")

def find_checkpoint(root, category):
    files = list((root / category).rglob("model.ckpt"))
    if not files:
        return None

    def version(path):
        for part in path.parts:
            if part.startswith("v") and part[1:].isdigit():
                return int(part[1:])
        return -1

    return max(files, key=version)

def predict(model, engine, folder, checkpoint):
    dataset = PredictDataset(path=folder, image_size=IMAGE_SIZE)

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(checkpoint),
        return_predictions=True,
    )

    return predictions

def benchmark(model_name, category, checkpoint, folder):
    model = (
        Patchcore()
        if model_name == "PatchCore Fast"
        else Dinomaly(
            encoder_name="vit_base_patch14_reg4_dinov2",
            decoder_depth=8,
        )
    )

    engine = Engine(
    accelerator=DEVICE,
    devices=1,
    default_root_dir=RESULTS / "benchmark_temp",
    logger=False,
)

    # Warm-up predictions are excluded from timing.
    warmup_folder = folder / "warmup"
    test_folder = folder / "timed"

    predict(model, engine, warmup_folder, checkpoint)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    predictions = predict(model, engine, test_folder, checkpoint)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    count = len(predictions) if predictions is not None else MAX_IMAGES

    del predictions, engine, model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return elapsed, count

def main():
    rows = []

    print("DEVICE:", DEVICE)

    for category in CATEGORIES:
        print("\nCATEGORY:", category.upper())

        test_folder = DATASET / category / "test"

        images = sorted(
            p for p in test_folder.rglob("*")
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
        )

        if len(images) < MAX_IMAGES + WARMUP_IMAGES:
            print("Not enough images. Skipping.")
            continue

        patch_ckpt = find_checkpoint(RESULTS / "Patchcore_Fast", category)
        dino_ckpt = find_checkpoint(RESULTS / "Dinomaly", category)

        if not patch_ckpt or not dino_ckpt:
            print("Checkpoint missing. Skipping.")
            continue

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            warmup_folder = base / "warmup"
            timed_folder = base / "timed"
            warmup_folder.mkdir()
            timed_folder.mkdir()

            for i, image in enumerate(images[:WARMUP_IMAGES]):
                shutil.copy2(image, warmup_folder / f"{i}_{image.name}")

            for i, image in enumerate(
                images[WARMUP_IMAGES:WARMUP_IMAGES + MAX_IMAGES]
            ):
                shutil.copy2(image, timed_folder / f"{i}_{image.name}")

            for name, ckpt in [
                ("PatchCore Fast", patch_ckpt),
                ("Dinomaly", dino_ckpt),
            ]:
                try:
                    elapsed, count = benchmark(
                        name, category, ckpt, base
                    )

                    row = {
                        "Category": category,
                        "Model": name,
                        "Images": count,
                        "Total_Time_Seconds": round(elapsed, 4),
                        "Average_Inference_ms": round(
                            elapsed * 1000 / count, 3
                        ),
                        "FPS": round(count / elapsed, 3),
                        "Checkpoint": str(ckpt),
                    }

                    rows.append(row)
                    print(row)

                    with open(OUTPUT, "w", newline="") as f:
                        writer = csv.DictWriter(
                            f, fieldnames=list(rows[0].keys())
                        )
                        writer.writeheader()
                        writer.writerows(rows)

                except Exception as error:
                    print("ERROR:", name, category, error)
                    raise

    print("\nBENCHMARK COMPLETED")
    print("CSV:", OUTPUT)

if __name__ == "__main__":
    main()