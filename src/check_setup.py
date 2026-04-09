import torch
import monai

print("PyTorch version: ", torch.__version__)
print("MONAI version: ", monai.__version__)
print("MPS Available: ", torch.backends.mps.is_available())