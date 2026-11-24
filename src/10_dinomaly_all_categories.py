
from pathlib import Path
import csv
import torch
import gc

from anomalib.models import Dinomaly
from anomalib.data import Folder
from anomalib.engine import Engine

# ==========================================
# CONFIGURATION
# ==========================================

DATASET_ROOT = Path("dataset")
RESULTS_DIR = Path("results")
CSV_FILE = RESULTS_DIR / "dinomaly_all_categories.csv"

BOTTLE_CHECKPOINT = Path(
    "results/Dinomaly/bottle/Dinomaly/bottle/"
    "v1/weights/lightning/model.ckpt"
)

categories = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

metrics_columns = [
    "Image_AUROC",
    "Image_F1Score",
    "Pixel_AUROC",
    "Pixel_F1Score",
]

# Map Anomalib's actual metric names to CSV columns
metric_mapping = {
    "Image_AUROC": "image_AUROC",
    "Image_F1Score": "image_F1Score",
    "Pixel_AUROC": "pixel_AUROC",
    "Pixel_F1Score": "pixel_F1Score",
}

RESULTS_DIR.mkdir(exist_ok=True)

device = "gpu" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("medium")

# ==========================================
# LOAD AND REPAIR EXISTING CSV
# ==========================================

all_results = []

if CSV_FILE.exists():
    with open(CSV_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)

        for old_row in reader:
            category = old_row.get("Category", "")

            # Convert previous lowercase columns if present
            row = {"Category": category}

            for column, actual_name in metric_mapping.items():
                value = old_row.get(column)

                if value in ("", None, "None"):
                    value = old_row.get(actual_name)

                row[column] = value if value not in ("", None, "None") else ""

            all_results.append(row)

# Remove duplicate category rows
unique = {}

for row in all_results:
    unique[row["Category"]] = row

all_results = list(unique.values())

def is_completed(row):
    return all(
        row.get(metric) not in ("", None, "None")
        for metric in metrics_columns
    )

completed = {
    row["Category"]
    for row in all_results
    if is_completed(row)
}

print("Using device:", device)
print("Already completed:", sorted(completed))

# ==========================================
# PROCESS CATEGORIES
# ==========================================

for category in categories:

    if category in completed:
        print("Skipping completed:", category)
        continue

    print("\n" + "=" * 50)
    print("CATEGORY:", category.upper())
    print("=" * 50)

    dataset_path = DATASET_ROOT / category
    test_path = dataset_path / "test"

    if not (dataset_path / "train" / "good").exists():
        print("Training folder missing:", category)
        continue

    if not (test_path / "good").exists():
        print("Test folder missing:", category)
        continue

    abnormal_dirs = [
        f"test/{folder.name}"
        for folder in test_path.iterdir()
        if folder.is_dir() and folder.name != "good"
    ]

    if not abnormal_dirs:
        print("No defect folders:", category)
        continue

    datamodule = None
    model = None
    engine = None

    try:
        datamodule = Folder(
            name=category,
            root=dataset_path,
            normal_dir="train/good",
            abnormal_dir=abnormal_dirs,
            normal_test_dir="test/good",
            mask_dir="ground_truth",
            normal_split_ratio=0.2,
            train_batch_size=1,
            eval_batch_size=1,
            num_workers=0,
        )

        model = Dinomaly(
            encoder_name="vit_base_patch14_reg4_dinov2",
            decoder_depth=8,
        )

        engine = Engine(
            accelerator=device,
            devices=1,
            max_epochs=1,
            default_root_dir=RESULTS_DIR / "Dinomaly" / category,
        )

        if category == "bottle":
            print("Loading existing bottle checkpoint...")

            test_results = engine.test(
                model=model,
                datamodule=datamodule,
                ckpt_path=str(BOTTLE_CHECKPOINT),
            )

        else:
            print("Training:", category)

            engine.fit(
                model=model,
                datamodule=datamodule,
            )

            print("Testing:", category)

            test_results = engine.test(
                model=model,
                datamodule=datamodule,
            )

        if not test_results:
            raise RuntimeError("No test metrics returned!")

        metrics = test_results[0]

        print("Raw metrics:", metrics)

        row = {"Category": category}

        for column, actual_name in metric_mapping.items():

            value = metrics.get(actual_name)

            if value is not None:
                if torch.is_tensor(value):
                    value = value.item()

                row[column] = float(value)
            else:
                row[column] = ""

        if not is_completed(row):
            raise RuntimeError(
                f"Some metrics are missing for {category}: {row}"
            )

        # Update category without duplicating it
        all_results = [
            r for r in all_results
            if r["Category"] != category
        ]

        all_results.append(row)

        # Save immediately after every category
        with open(CSV_FILE, "w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["Category"] + metrics_columns,
            )

            writer.writeheader()
            writer.writerows(all_results)

        completed.add(category)

        print("\nSUCCESS:", category)
        print("Metrics:", row)
        print("CSV saved:", CSV_FILE)

    except Exception as error:
        print("\nERROR:", category)
        print(error)
        print("Stopping. Rerun this script after fixing the error.")
        break

    finally:
        if engine is not None:
            del engine

        if model is not None:
            del model

        if datamodule is not None:
            del datamodule

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "=" * 50)
print("DINOMALY PROCESS FINISHED")
print("=" * 50)

print("Completed:", len(completed), "/", len(categories))
print("CSV:", CSV_FILE)

for row in all_results:
    print(row)