
import torch

# Select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# Create a large matrix on GPU
a = torch.randn(3000, 3000, device=device)
b = torch.randn(3000, 3000, device=device)

# Perform matrix multiplication
c = torch.matmul(a, b)

# Wait for GPU calculation to finish
torch.cuda.synchronize()

print("Calculation completed!")
print("Result shape:", c.shape)
print("GPU name:", torch.cuda.get_device_name(0))