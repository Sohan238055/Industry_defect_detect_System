from pathlib import Path

# Get the dataset folder location
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"

# Check whether dataset exists
if not DATASET_DIR.exists():
    print("Dataset folder not found!")
else:
    print("Dataset found successfully!")

    # Find all valid categories
    categories = []

    for folder in DATASET_DIR.iterdir():
        if folder.is_dir() and (folder / "train" / "good").exists():
            categories.append(folder.name)

    print("\nTotal categories:", len(categories))
    print("\nAvailable categories:")

    for category in sorted(categories):
        print(category)
