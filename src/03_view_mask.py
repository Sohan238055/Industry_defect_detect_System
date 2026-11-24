
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"

# Select tile category
tile_path = DATASET_DIR / "tile"

# Select a defective image
defect_folders = [
    folder for folder in (tile_path / "test").iterdir()
    if folder.is_dir() and folder.name != "good"
]

defect_folder = defect_folders[0]
image_path = next(defect_folder.glob("*.png"))

# Find corresponding ground-truth mask
mask_path = (
    tile_path / "ground_truth" /
    defect_folder.name / f"{image_path.stem}_mask.png"
)

# Read original image and mask
image = cv2.imread(str(image_path))
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

if image is None or mask is None:
    raise FileNotFoundError("Could not load image or mask. Check the paths.")

# Convert original image to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display both
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Defective Tile")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="gray")
plt.title("Ground Truth Mask")
plt.axis("off")

plt.show()
