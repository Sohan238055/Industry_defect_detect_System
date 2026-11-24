
from pathlib import Path
import torch

from anomalib.models import Dinomaly
from anomalib.data import Folder
from anomalib.engine import Engine

category = "bottle"

dataset_path = Path("dataset") / category
test_path = dataset_path / "test"

# Automatically detect actual defect folders
abnormal_dirs = [
    f"test/{folder.name}"
    for folder in test_path.iterdir()
    if folder.is_dir() and folder.name != "good"
]

print("Category:", category)
print("Defect folders:", abnormal_dirs)

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
# 3. Create Vision Transformer model
model = Dinomaly(
    encoder_name="vit_base_patch14_reg4_dinov2",
    decoder_depth=8,
)

print("Dinomaly model created!")

# 4. Configure engine
engine = Engine(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    max_epochs=1,
    default_root_dir="results/Dinomaly/bottle",
)

# 5. Train
print("Starting Dinomaly training...")

engine.fit(
    model=model,
    datamodule=datamodule,
)

print("Dinomaly training completed!")

# 6. Test
print("Starting Dinomaly testing...")

test_results = engine.test(
    model=model,
    datamodule=datamodule,
)

print("Dinomaly testing completed!")
print("Results:", test_results)
