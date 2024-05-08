import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))

# Kiểm tra nếu CUDA có sẵn
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Running on device: {device}")
else:
    print("CUDA is not available")

# Tạo một tensor đơn giản trên GPU
x = torch.rand(3, 3).to(device)
print(x)

# Kiểm tra nếu CUDA khả dụng
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Tạo tensor đơn giản trên GPU
x = torch.rand((100, 100), device=device)
print(x)
