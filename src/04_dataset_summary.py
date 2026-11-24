
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"

categories = [
    folder for folder in DATASET_DIR.iterdir()
    if folder.is_dir() and (folder / "train" / "good").exists()
]

print("DATASET SUMMARY")
print("-" * 65)

for category in sorted(categories):

    train_good = len(list(
        (category / "train" / "good").glob("*.png")
    ))

    test_good = len(list(
        (category / "test" / "good").glob("*.png")
    ))

    defect_count = 0

    for folder in (category / "test").iterdir():
        if folder.is_dir() and folder.name != "good":
            defect_count += len(list(folder.glob("*.png")))

    print("\nCategory:", category.name)
    print("Training normal images:", train_good)
    print("Testing normal images:", test_good)
    print("Testing defective images:", defect_count)

print("\nTotal categories:", len(categories))