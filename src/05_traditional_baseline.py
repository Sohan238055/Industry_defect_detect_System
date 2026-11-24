
import cv2
import numpy as np
from pathlib import Path

from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

# Dataset location
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"

CATEGORY = "tile"
category_path = DATASET_DIR / CATEGORY


# Extract GLCM and Canny features
def extract_features(image_path):

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image = cv2.resize(image, (128, 128))

    # Convert grayscale values into 8 levels
    quantized = (image // 32).astype(np.uint8)

    # GLCM texture features
    glcm = graycomatrix(
        quantized,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=8,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, "contrast").mean()
    homogeneity = graycoprops(glcm, "homogeneity").mean()
    energy = graycoprops(glcm, "energy").mean()
    correlation = graycoprops(glcm, "correlation").mean()

    # Canny edge detection
    edges = cv2.Canny(image, 100, 200)
    edge_density = np.mean(edges > 0)

    return [
        contrast,
        homogeneity,
        energy,
        correlation,
        edge_density
    ]


# Get training images (normal only)
train_folder = category_path / "train" / "good"
train_images = sorted(train_folder.glob("*.png"))

X_train = np.array([
    extract_features(path) for path in train_images
])

# Get normal test images
test_good = sorted(
    (category_path / "test" / "good").glob("*.png")
)

# Get defective test images
test_defect = []

for folder in (category_path / "test").iterdir():
    if folder.is_dir() and folder.name != "good":
        test_defect.extend(sorted(folder.glob("*.png")))

# Combine test images
test_images = test_good + test_defect

X_test = np.array([
    extract_features(path) for path in test_images
])

# Labels: 0 = Normal, 1 = Defective
y_test = np.array(
    [0] * len(test_good) + [1] * len(test_defect)
)

print("Training images:", len(X_train))
print("Testing images:", len(X_test))
print("Feature count:", X_train.shape[1])

# Train Isolation Forest using normal images only
model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42
)

model.fit(X_train)

# Higher score means more anomalous
anomaly_scores = -model.decision_function(X_test)

# Evaluate image-level AUROC
auc = roc_auc_score(y_test, anomaly_scores)

print("\nTraditional Model Results")
print("-------------------------")
print("Image-level AUROC:", round(auc, 4))
