
from pathlib import Path
import csv
import torch

from anomalib.models import Patchcore
from anomalib.data import Folder
from anomalib.engine import Engine


# 1. Dataset location
dataset_root = Path("dataset")

categories = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

# 2. CSV output location
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

csv_file = results_dir / "patchcore_all_categories.csv"

all_results = []

# 3. Check GPU availability
device = "gpu" if torch.cuda.is_available() else "cpu"

print("Using device:", device)
print("Starting PatchCore evaluation on 15 categories...")


# 4. Train and test each category
for category in categories:

    print("\n" + "=" * 60)
    print("CATEGORY:", category.upper())
    print("=" * 60)

    dataset_path = dataset_root / category
    test_path = dataset_path / "test"

    if not dataset_path.exists():
        print("Dataset not found. Skipping:", dataset_path)
        continue

    if not test_path.exists():
        print("Test folder not found. Skipping:", test_path)
        continue

    # Automatically find defect folders
    abnormal_dirs = [
        f"test/{folder.name}"
        for folder in test_path.iterdir()
        if folder.is_dir() and folder.name != "good"
    ]

    if not abnormal_dirs:
        print("No defect folders found. Skipping:", category)
        continue

    # 5. Load category dataset
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

    # 6. Create PatchCore model
    model = Patchcore()

    # 7. Configure engine
    engine = Engine(
        accelerator=device,
        devices=1,
        max_epochs=1,
        default_root_dir=results_dir / "Patchcore" / category,
    )

    # 8. Train
    print("Starting training:", category)

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    print("Training completed:", category)

    # 9. Test
    print("Starting testing:", category)

    test_results = engine.test(
        model=model,
        datamodule=datamodule,
    )

    print("Testing completed:", category)

    # 10. Collect metrics
    row = {
        "Category": category,
        "Image_AUROC": None,
        "Image_F1Score": None,
        "Pixel_AUROC": None,
        "Pixel_F1Score": None,
    }

    if test_results:
        metrics = test_results[0]

        for key in row:
            if key in metrics:
                value = metrics[key]

                if torch.is_tensor(value):
                    value = value.item()

                if key != "Category":
                    row[key] = float(value)

    all_results.append(row)

    print("\nCategory results:")
    print(row)


# 11. Save results to CSV
if all_results:

    fieldnames = [
        "Category",
        "Image_AUROC",
        "Image_F1Score",
        "Pixel_AUROC",
        "Pixel_F1Score",
    ]

    with open(csv_file, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 60)
    print("ALL CATEGORY EVALUATIONS COMPLETED!")
    print("=" * 60)

    print("Results saved to:", csv_file)

    for row in all_results:
        print(row)

else:
    print("No category results were generated.")
