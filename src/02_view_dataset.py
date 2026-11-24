import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Dataset location
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"

# Start with tile category
CATEGORY = "tile"

category_path = DATASET_DIR / CATEGORY

# Get one normal image
normal_path = next((category_path / "train" / "good").glob("*.png"))

# Get one defective image
defect_folders = [
    folder for folder in (category_path / "test").iterdir()
    if folder.is_dir() and folder.name != "good"
]

defect_folder = defect_folders[0]
defect_path = next(defect_folder.glob("*.png"))

# Read images
normal_image = cv2.imread(str(normal_path))
defect_image = cv2.imread(str(defect_path))

# Convert BGR to RGB
normal_image = cv2.cvtColor(normal_image, cv2.COLOR_BGR2RGB)
defect_image = cv2.cvtColor(defect_image, cv2.COLOR_BGR2RGB)

# Display images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(normal_image)
plt.title("Normal Tile")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(defect_image)
plt.title("Defective Tile: " + defect_folder.name)
plt.axis("off")

plt.show()