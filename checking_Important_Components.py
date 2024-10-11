import torch
import os
import subprocess
from torch.utils.cpp_extension import CUDA_HOME


def get_vram_info():
    """
    Trích xuất thông tin VRAM hiện có và điều chỉnh giá trị
    N_GPU_LAYERS và N_BATCH dựa vào VRAM.
    """
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_vram = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # Chuyển đổi từ byte sang GB
            print(f"Total VRAM (GB): {total_vram:.2f}")

            # Điều chỉnh N_GPU_LAYERS và N_BATCH dựa vào dung lượng VRAM
            if total_vram >= 40:  # VRAM lớn hơn hoặc bằng 40GB
                number_of_gpu_layers = 70  # Để cho phép mô hình lớn với nhiều layers hơn
                number_batch = 1024
            elif total_vram >= 24:  # VRAM từ 24GB trở lên
                number_of_gpu_layers = 50
                number_batch = 512
            elif total_vram >= 16:  # VRAM từ 16GB trở lên
                number_of_gpu_layers = 35
                number_batch = 512
            elif total_vram >= 8:  # VRAM từ 8GB trở lên
                number_of_gpu_layers = 20
                number_batch = 256
            else:  # VRAM dưới 8GB
                number_of_gpu_layers = 12
                number_batch = 128

            # In thông số ra để kiểm tra
            print(f"N_GPU_LAYERS: {number_of_gpu_layers}, N_BATCH: {number_batch}")

            return number_of_gpu_layers, number_batch
        else:
            print("CUDA không khả dụng.")
            return None, None
    except Exception as cuda_error:
        print(f"Có lỗi xảy ra: {cuda_error}")
        return None, None


# Kiểm tra CUDA và lấy thông số VRAM
print("Torch Version:", torch.__version__)
print("Is CUDA available?", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("Number of GPUs available:", torch.cuda.device_count())

# Kiểm tra đường dẫn mà CUDA của torch đang sử dụng
cuda_home = torch.utils.cpp_extension.CUDA_HOME
print(f"CUDA Home (Torch is using): {cuda_home}")

# Kiểm tra CUDA có trong môi trường conda không
conda_cuda_path = os.environ.get("CONDA_PREFIX")
print(f"CONDA_PREFIX: {conda_cuda_path}")

# Kiểm tra CUDA từ conda environment
if conda_cuda_path and os.path.exists(os.path.join(conda_cuda_path, "Library", "bin", "cudart64_110.dll")):
    print(f"CUDA đang được sử dụng từ Conda Environment: {conda_cuda_path}")
else:
    print("CUDA không được cài đặt trong môi trường Conda hoặc không tìm thấy cudart64_110.dll")

# Kiểm tra phiên bản CUDA Toolkit trong môi trường conda
try:
    conda_cuda_version = subprocess.check_output("conda list cudatoolkit", shell=True).decode()
    print("CUDA Toolkit version in conda environment:\n", conda_cuda_version)
except Exception as e:
    print("Không thể lấy phiên bản CUDA Toolkit trong môi trường conda:", e)

# Kiểm tra CUDA toolkit của hệ điều hành (host)
try:
    nvcc_version = subprocess.check_output("nvcc --version", shell=True).decode()
    print("CUDA Toolkit version on host system:\n", nvcc_version)
except Exception as e:
    print("Không thể lấy phiên bản CUDA Toolkit của hệ điều hành (host):", e)

# Thử tạo tensor
try:
    if torch.cuda.is_available():
        x = torch.rand(5, 3).cuda()  # Tạo một tensor ngẫu nhiên trên GPU
        print("Tensor đã được tạo thành công trên GPU:")
        print(x)

        # Lấy ra N_GPU_LAYERS và N_BATCH dựa trên VRAM
        n_gpu_layers, n_batch = get_vram_info()
        print(f"Thông số điều chỉnh: N_GPU_LAYERS = {n_gpu_layers}, N_BATCH = {n_batch}")
    else:
        # Trường hợp CUDA không khả dụng, tạo tensor trên CPU
        x = torch.rand(5, 3)  # Tạo tensor trên CPU
        print("CUDA không khả dụng. Tensor đã được tạo trên CPU:")
        print(x)
except Exception as e:
    print(f"Có lỗi xảy ra: {e}")

# Lấy danh sách các dependencies từ pip và lưu vào file
try:
    pip_requirements = subprocess.check_output("pip freeze", shell=True).decode()
    with open("pip_requirements.txt", "w") as f:
        f.write(pip_requirements)
    print("Danh sách các dependencies từ pip đã được ghi vào pip_requirements.txt")
except Exception as e:
    print(f"Không thể lấy danh sách dependencies từ pip: {e}")

# Lấy danh sách các dependencies từ conda và lưu vào file
try:
    conda_requirements = subprocess.check_output("conda list", shell=True).decode()
    with open("conda_requirements.txt", "w") as f:
        f.write(conda_requirements)
    print("Danh sách các dependencies từ conda đã được ghi vào conda_requirements.txt")
except Exception as e:
    print(f"Không thể lấy danh sách dependencies từ conda: {e}")
