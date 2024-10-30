import torch

# Kiểm tra xem có GPU không
if torch.cuda.is_available():
    print("CUDA is available. You can use GPU.")
    print("Number of GPUs available:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Using CPU.")
