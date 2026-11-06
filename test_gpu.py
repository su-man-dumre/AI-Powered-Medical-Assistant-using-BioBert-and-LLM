
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda")

    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory:", round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
    ), "GB")

else:
    device = torch.device("cpu")
    print("Using CPU")

# Test calculation
x = torch.rand(1000, 1000).to(device)
y = x @ x

print("Calculation device:", y.device)
