
from pathlib import Path
import csv
import torch

from anomalib.models import Patchcore
from anomalib.data import Folder
from anomalib.engine import Engine

# 1. Dataset categories
dataset_root = Path("dataset")

categories = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

# 2. Create results folder
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

csv_file = results_dir / "patchcore_fast_results.csv"

fieldnames = [
    "Category",
    "Image_AUROC",
    "Image_F1Score",
    "Pixel_AUROC",
    "Pixel_F1Score"
]

# 3. Load previous results, if available
completed = {}

if csv_file.exists():
    with open(csv_file, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if all(row.get(key) not in ("", None) for key in fieldnames[1:]):
                completed[row["Category"]] = row

# 4. Select GPU
device = "gpu" if torch.cuda.is_available() else "cpu"

print("Using device:", device)
print("Already completed:", list(completed.keys()))

# 5. Process all categories
for category in categories:

    if category in completed:
        print("Skipping completed:", category)
        continue

    print("\n" + "=" * 50)
    print("CATEGORY:", category.upper())
    print("=" * 50)

    dataset_path = dataset_root / category
    test_path = dataset_path / "test"

    if not test_path.exists():
        print("Test folder missing. Skipping:", category)
        continue

    abnormal_dirs = [
        f"test/{folder.name}"
        for folder in test_path.iterdir()
        if folder.is_dir() and folder.name != "good"
    ]

    if not abnormal_dirs:
        print("No defect folders. Skipping:", category)
        continue

    # 6. Load category dataset
    datamodule = Folder(
        name=category,
        root=dataset_path,
        normal_dir="train/good",
        abnormal_dir=abnormal_dirs,
        normal_test_dir="test/good",
        mask_dir="ground_truth",
        normal_split_ratio=0.2,
        train_batch_size=4,
        eval_batch_size=4,
        num_workers=0,
    )

    # 7. Create PatchCore model
    model = Patchcore(
        coreset_sampling_ratio=0.01
    )

    # 8. Create engine
    engine = Engine(
        accelerator=device,
        devices=1,
        max_epochs=1,
        default_root_dir=results_dir / "Patchcore_Fast" / category,
    )

    # 9. Train
    print("Training:", category)
    engine.fit(model=model, datamodule=datamodule)

    # 10. Test
    print("Testing:", category)
    test_results = engine.test(
        model=model,
        datamodule=datamodule
    )

    # 11. Extract metrics
    row = {
        "Category": category,
        "Image_AUROC": "",
        "Image_F1Score": "",
        "Pixel_AUROC": "",
        "Pixel_F1Score": "",
    }

    if test_results:
        metrics = test_results[0]

        mapping = {
            "Image_AUROC": "image_AUROC",
            "Image_F1Score": "image_F1Score",
            "Pixel_AUROC": "pixel_AUROC",
            "Pixel_F1Score": "pixel_F1Score",
        }

        for csv_key, metric_key in mapping.items():
            value = metrics.get(metric_key)

            if value is not None:
                if torch.is_tensor(value):
                    value = value.item()

                row[csv_key] = float(value)

    # 12. Save after every category
    completed[category] = row

    with open(csv_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(completed.values())

    print("Completed:", category)
    print("Metrics:", row)

print("\nALL CATEGORY PROCESSING FINISHED!")
print("Results saved at:", csv_file)