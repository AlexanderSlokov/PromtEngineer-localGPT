import torch
print("Torch Version:", torch.__version__)
print("CUDA Version:", torch.version.cuda)
print("Is CUDA available?", torch.cuda.is_available())
print("Number of GPUs available:", torch.cuda.device_count())

try:
    # Kiểm tra xem CUDA có khả dụng không
    if torch.cuda.is_available():
        x = torch.rand(5, 3).cuda()  # Tạo một tensor ngẫu nhiên trên GPU
        print("Tensor đã được tạo thành công trên GPU:")
        print(x)
    else:
        # Trường hợp CUDA không khả dụng, tạo tensor trên CPU
        x = torch.rand(5, 3)  # Tạo tensor trên CPU
        print("CUDA không khả dụng. Tensor đã được tạo trên CPU:")
        print(x)
except Exception as e:
    print(f"Có lỗi xảy ra: {e}")

