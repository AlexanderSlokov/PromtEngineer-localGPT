Để bắt đầu sử dụng LocalGPT sau khi bạn đã cài đặt các dependencies và PyTorch theo yêu cầu, bạn có thể làm theo các bước sau:

### 1. **Ingest dữ liệu của bạn**
Đầu tiên, bạn cần ingest (đưa vào hệ thống) các tài liệu mà bạn muốn sử dụng với LocalGPT.

Nếu bạn có GPU CUDA và đã cấu hình đúng, chạy lệnh sau để ingest dữ liệu:

```bash
python ingest.py --device_type cuda
```

Nếu bạn muốn chạy trên CPU hoặc M1/M2, hãy chỉ định loại thiết bị:

- Chạy trên CPU:
    ```bash
    python ingest.py --device_type cpu
    ```

- Chạy trên M1/M2 (Apple Silicon):
    ```bash
    python ingest.py --device_type mps
    ```

### 2. **Chạy LocalGPT để giao tiếp với tài liệu của bạn**

Sau khi ingest xong tài liệu, bạn có thể chạy `run_localGPT.py` để bắt đầu tương tác với các tài liệu đã được lưu trữ:

```bash
python run_localGPT.py --device_type cuda
```

Nếu bạn muốn chạy trên CPU hoặc M1/M2, sử dụng các tùy chọn như trong bước 1:

- Chạy trên CPU:
    ```bash
    python run_localGPT.py --device_type cpu
    ```

- Chạy trên M1/M2:
    ```bash
    python run_localGPT.py --device_type mps
    ```

### 3. **Kiểm tra xem model và tài liệu đã được tải thành công chưa**
Trong quá trình chạy `run_localGPT.py`, bạn sẽ thấy thông báo cho biết model và vector store đã được tải thành công hay không. Nếu mọi thứ đều ổn, bạn sẽ được hiển thị một prompt như sau:

```bash
> Enter a query:
```

Tại đây, bạn có thể nhập câu hỏi liên quan đến tài liệu của mình và LocalGPT sẽ trả lời dựa trên ngữ cảnh của tài liệu đã ingest.

### 4. **Sử dụng các tùy chọn bổ sung**
Nếu muốn xem những đoạn tài liệu nào đã được lấy ra để tạo câu trả lời, bạn có thể sử dụng tùy chọn `--show_sources`:

```bash
python run_localGPT.py --show_sources
```

Hoặc nếu muốn lưu lại lịch sử câu hỏi và câu trả lời, hãy sử dụng tùy chọn `--save_qa`:

```bash
python run_localGPT.py --save_qa
```

Lịch sử sẽ được lưu tại tệp `/local_chat_history/qa_log.csv`.

### 5. **Sử dụng Giao diện đồ họa (GUI)**
Nếu bạn muốn sử dụng giao diện đồ họa để tương tác với LocalGPT, bạn có thể chạy `run_localGPT_API.py` trước để khởi động API và sau đó chạy `localGPTUI.py` để mở giao diện:

1. Chạy API:
   ```bash
   python run_localGPT_API.py
   ```

2. Chạy GUI:
   ```bash
   python localGPTUI.py
   ```

3. Mở trình duyệt và truy cập `http://localhost:5111/` để bắt đầu sử dụng.

### 6. **Lưu ý**
- Khi chạy LocalGPT lần đầu, chương trình sẽ cần tải model từ internet (ví dụ như `TheBloke/Llama-2-7b-Chat-GGUF`), do đó bạn cần kết nối mạng. Sau khi tải xong, bạn có thể sử dụng mà không cần kết nối internet.
- Nếu gặp bất kỳ lỗi nào liên quan đến CUDA hoặc GPU không tương thích, hãy kiểm tra lại cấu hình CUDA và GPU của bạn bằng cách chạy các lệnh như `nvcc --version` hoặc `nvidia-smi`.

Bây giờ bạn đã sẵn sàng để sử dụng LocalGPT cho các tương tác bảo mật và cục bộ với tài liệu của mình! Nếu có bất kỳ thắc mắc nào thêm, hãy chia sẻ để mình hỗ trợ tiếp nhé!
