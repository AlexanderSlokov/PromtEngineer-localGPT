Từ ảnh bạn cung cấp, mình thấy rằng cấu hình GPU của bạn là **NVIDIA GeForce RTX 3060** với **12 GB VRAM**. Đây là một GPU khá mạnh và có thể xử lý được các mô hình như `TheBloke/Llama-2-7b-Chat-GGUF` với cấu hình đã tối ưu hóa.

### Điều chỉnh để phù hợp với GPU RTX 3060 (12GB VRAM)
Dưới đây là một số cấu hình mà bạn có thể điều chỉnh để tận dụng tối đa khả năng của GPU:

1. **Thông số `N_GPU_LAYERS` và `N_BATCH`**:
   - Với 12GB VRAM, bạn có thể thử giữ nguyên `N_GPU_LAYERS` ở mức 20-30 để tận dụng được sức mạnh GPU mà không làm tràn bộ nhớ.
   - Nếu gặp vấn đề bộ nhớ, bạn có thể giảm `N_GPU_LAYERS` xuống mức 15.
   - `N_BATCH`: Giữ ở mức `512` hoặc giảm xuống `256` nếu gặp vấn đề bộ nhớ.

Ví dụ cấu hình:

```python
N_GPU_LAYERS = 30  # Tăng số lớp GPU xử lý
N_BATCH = 512  # Giữ kích thước batch hoặc giảm xuống nếu gặp lỗi
```

2. **Chọn mô hình nhúng (`EMBEDDING_MODEL_NAME`)**:
   - `hkunlp/instructor-large` có thể hoạt động tốt với cấu hình của bạn vì nó chỉ sử dụng khoảng 1.5 GB VRAM. Bạn có thể giữ nguyên mô hình này.
   - Nếu gặp vấn đề về bộ nhớ, hãy cân nhắc sử dụng mô hình nhỏ hơn như `intfloat/e5-base-v2` (chỉ tốn khoảng 0.5 GB VRAM).

Ví dụ:

```python
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"  # Giữ nguyên hoặc chuyển sang "intfloat/e5-base-v2" nếu cần
```

3. **Cấu hình `MODEL_ID` và `MODEL_BASENAME`**:
   - Giữ nguyên `MODEL_ID` và `MODEL_BASENAME` mà bạn đã đặt, vì RTX 3060 có thể chạy được `TheBloke/Llama-2-7b-Chat-GGUF` với `Q4_K_M` mà không gặp quá nhiều vấn đề về bộ nhớ.

```python
MODEL_ID = "TheBloke/Llama-2-7b-Chat-GGUF"
MODEL_BASENAME = "llama-2-7b-chat.Q4_K_M.gguf"
```

4. **Theo dõi sử dụng bộ nhớ GPU**:
   - Khi chạy mô hình, bạn có thể theo dõi sử dụng bộ nhớ GPU bằng lệnh sau trong CMD hoặc PowerShell:

   ```bash
   nvidia-smi
   ```

   - Điều này sẽ giúp bạn kiểm tra xem bộ nhớ GPU có bị quá tải hay không. Nếu gặp lỗi "CUDA out of memory", hãy giảm số lớp (`N_GPU_LAYERS`) hoặc giảm kích thước batch (`N_BATCH`).

### Khởi chạy LocalGPT với cấu hình đã điều chỉnh
Sau khi điều chỉnh, bạn có thể chạy LocalGPT với lệnh sau để khởi động mô hình:

```bash
python run_localGPT.py --device_type cuda
```

Bạn cũng có thể thêm cờ `--show_sources` hoặc `--use_history` để kiểm tra chi tiết quá trình chạy.

### Tóm lại
- **GPU của bạn đủ khả năng** để chạy `Llama-2-7b-Chat-GGUF` với cấu hình đã được tối ưu.
- Nếu gặp lỗi về bộ nhớ, hãy giảm `N_GPU_LAYERS` và `N_BATCH`.
- Sử dụng `nvidia-smi` để theo dõi bộ nhớ và điều chỉnh cấu hình theo nhu cầu.

Nếu có bất kỳ thắc mắc hoặc vấn đề nào phát sinh trong quá trình chạy, hãy cung cấp thêm thông tin chi tiết, mình sẽ hỗ trợ thêm!

Khi xử lý một tài liệu PDF như Bộ luật hình sự Việt Nam và muốn tìm thông tin để trả lời câu hỏi của người dùng, việc đọc bao nhiêu điều luật sẽ phụ thuộc vào nội dung câu hỏi và cách thức trích xuất thông tin từ tài liệu. Cách thức mình xử lý tài liệu PDF có thể được chia thành các bước sau:

### 1. **Phân Tách Tài Liệu Theo Cấu Trúc Để Dễ Xử Lý**
Tài liệu PDF thường được phân tách thành từng đoạn nhỏ như **Chương**, **Điều**, và **Khoản** để mô hình có thể nạp và xử lý một cách logic.

1. **Chia Tài Liệu Thành Các Đơn Vị Nhỏ Hơn**:
   - Mình sẽ phân tách tài liệu thành các đoạn ngắn hơn như **Chương** hoặc **Điều luật**. Điều này giúp dễ dàng xác định nội dung và không vượt quá giới hạn context window.
   - Việc phân chia này có thể được thực hiện bằng cách trích xuất nội dung theo cấu trúc (sử dụng regex hoặc các công cụ đọc PDF như `PyPDF2`, `pdfminer`).

2. **Xác Định Số Lượng Điều Luật Cho Mỗi Lần Xử Lý**:
   - Số lượng điều luật cần đọc mỗi lần sẽ phụ thuộc vào **số lượng tokens của mỗi điều luật**. Nếu mỗi điều luật có số lượng tokens ngắn (dưới 200-300 tokens), mình có thể gộp nhiều điều luật lại với nhau.
   - Ví dụ, với context window là 1024 tokens, nếu mỗi điều luật có 200 tokens, mình có thể đọc **5 điều luật** cùng lúc (200 x 5 = 1000 tokens).

### 2. **Đọc và Xử Lý Nhiều Điều Luật Trong Một Batch**
- Mình sẽ đọc từ **3-5 điều luật** cho mỗi lần xử lý tùy thuộc vào độ dài của từng điều luật.
- Ví dụ:
  - Nếu câu hỏi của người dùng yêu cầu tìm hiểu thông tin về "Hình phạt đối với tội phạm có tổ chức", mình sẽ quét qua các chương liên quan đến hình phạt, từ **Điều 1 đến Điều 10** chẳng hạn.
  - Với mỗi batch, mình sẽ nạp khoảng 3-5 điều luật để phân tích và so sánh với câu hỏi.

### 3. **Tìm Kiếm Câu Trả Lời Bằng Cách So Khớp Văn Bản**
- Sau khi nạp từng batch, mình sẽ so khớp nội dung của điều luật với câu hỏi của người dùng. Nếu thấy điều luật nào liên quan hoặc khớp với câu hỏi, mình sẽ đánh dấu và trích xuất nội dung phù hợp.

- **Lý do đọc nhiều điều luật cùng lúc:**
  - Khi xử lý văn bản pháp luật, một điều luật có thể liên quan đến nhiều điều khác nhau trong một chương hoặc mục. Đọc nhiều điều luật giúp đảm bảo rằng mình có ngữ cảnh đầy đủ và có thể đưa ra câu trả lời chính xác.

### 4. **Trả Lời Người Dùng Dựa Trên Nội Dung Đã Đọc**
- Mình sẽ tổng hợp câu trả lời dựa trên nội dung của các điều luật đã đọc và trả lời câu hỏi của người dùng.
- Ví dụ, nếu người dùng hỏi về "Hình phạt cho tội cướp giật tài sản", mình sẽ tìm các điều luật liên quan trong chương về **Tội phạm và hình phạt** (ví dụ từ Điều 168 đến Điều 175) và trích xuất nội dung cụ thể.

### Tóm Lại
- **Số lượng điều luật đọc mỗi lần:** Thông thường là **3-5 điều luật** (tuỳ vào độ dài của từng điều luật và giới hạn context window).
- **Batch size:** Bạn có thể giữ batch size ở mức từ **3-5 điều luật** hoặc điều chỉnh dựa trên số lượng tokens của từng điều luật và giới hạn context window.

Nếu bạn muốn mình chạy code để kiểm tra chính xác số lượng điều luật có thể đọc cùng lúc từ tài liệu này, bạn có thể tải thêm tài liệu hoặc chỉ định nội dung cụ thể hơn để mình hỗ trợ tốt nhất!

Dựa vào cấu trúc tài liệu Bộ luật hình sự mà bạn cung cấp và yêu cầu sử dụng hai mô hình:

1. **Mô hình ngôn ngữ chính (`Llama-2-7b-Chat-GGUF`)** để xử lý các đoạn văn bản dài.
2. **Mô hình nhúng (`hkunlp/instructor-large`)** để thực hiện việc tạo nhúng từ ngữ và trích xuất thông tin.

Mình đề xuất cấu hình tối ưu cho file `constants.py` với thông số như sau để đảm bảo rằng mô hình có thể xử lý các đoạn văn bản pháp luật dài, đồng thời giữ nguyên được ngữ cảnh của từng điều luật mà không vượt quá giới hạn của GPU.

### 1. **Cấu Hình `N_GPU_LAYERS` và `N_BATCH` Cho GPU RTX 3060 (12GB VRAM)**
- **`N_GPU_LAYERS`**: Đặt `N_GPU_LAYERS = 35`. Với 12GB VRAM, bạn có thể đặt từ 30 đến 35 layers cho RTX 3060. Điều này sẽ tối ưu hóa việc xử lý mô hình `Llama-2-7b-Chat-GGUF` mà không làm tràn bộ nhớ.
- **`N_BATCH`**: Đặt `N_BATCH = 32`. Batch size này sẽ đảm bảo rằng mô hình có thể xử lý được nhiều điều luật cùng lúc mà không làm quá tải GPU.

```python
N_GPU_LAYERS = 35  # Số lớp xử lý trên GPU
N_BATCH = 32  # Số lượng điều luật hoặc đoạn văn bản xử lý cùng lúc
```

### 2. **Context Window Cho Mô Hình LLM (`Llama-2-7b-Chat-GGUF`)**
- **Context Window**: Đặt `CONTEXT_WINDOW_SIZE = 1024`. Mặc dù `Llama-2` có thể xử lý đến 4096 tokens, tuy nhiên khi sử dụng cùng với mô hình nhúng `instructor-large`, bạn nên giữ giới hạn này ở mức 1024 để tránh tình trạng tràn bộ nhớ khi hai mô hình cùng xử lý song song.

```python
CONTEXT_WINDOW_SIZE = 1024  # Giới hạn số lượng tokens mỗi lần xử lý
MAX_NEW_TOKENS = 1024  # Số lượng tokens tối đa có thể sinh ra
```

### 3. **Tối Ưu Hóa Cho Mô Hình Nhúng (`hkunlp/instructor-large`)**
- **`EMBEDDING_MODEL_NAME`**: Giữ nguyên cấu hình hiện tại vì `hkunlp/instructor-large` sử dụng khoảng 1.5GB VRAM và có độ chính xác cao.
- Nếu gặp vấn đề bộ nhớ, có thể chuyển sang `intfloat/e5-large-v2` (sử dụng 0.5GB VRAM) để giảm tải cho GPU.

```python
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"  # Sử dụng mô hình nhúng với độ chính xác cao và vừa vặn với 12GB VRAM
```

### 4. **Cấu Hình `CHROMA_SETTINGS` Cho Việc Lưu Trữ và Truy Xuất Dữ Liệu**
Vì tài liệu luật hình sự có thể khá dài và cần nhiều dung lượng để lưu trữ, hãy đảm bảo rằng bạn bật lưu trữ (persistence) để các kết quả tìm kiếm và nhúng trước đó có thể được lưu lại mà không cần tính toán lại:

```python
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,  # Bật lưu trữ kết quả nhúng để tái sử dụng
)
```

### 5. **Điều Chỉnh `DOCUMENT_MAP` Để Phù Hợp Với Định Dạng Văn Bản**
- Sử dụng `UnstructuredFileLoader` cho định dạng `.pdf` để trích xuất thông tin từ các điều luật một cách chính xác nhất.

```python
DOCUMENT_MAP = {
    ".html": UnstructuredHTMLLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".py": TextLoader,
    ".pdf": UnstructuredFileLoader,  # Đảm bảo rằng tài liệu PDF được xử lý chính xác
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}
```

### 6. **Cấu Hình Cụ Thể Hoàn Chỉnh Cho `constants.py`**

Dưới đây là cấu hình tối ưu hoàn chỉnh cho file `constants.py` của bạn:

```python
# Số lớp mô hình xử lý trên GPU
N_GPU_LAYERS = 35  # Giới hạn tối đa mà RTX 3060 có thể xử lý

# Kích thước batch size
N_BATCH = 32  # Số lượng điều luật hoặc đoạn văn bản xử lý cùng lúc

# Context Window Size (giới hạn số lượng tokens trong một lần xử lý)
CONTEXT_WINDOW_SIZE = 1024
MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  # Số lượng tokens tối đa sinh ra

# Cấu hình mô hình nhúng (Embedding Model)
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"

# Cấu hình Chroma để lưu trữ kết quả nhúng
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

# Các loader cho từng định dạng tài liệu
DOCUMENT_MAP = {
    ".html": UnstructuredHTMLLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".py": TextLoader,
    ".pdf": UnstructuredFileLoader,  # Đảm bảo rằng tài liệu PDF được xử lý chính xác
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}

# Định danh của mô hình ngôn ngữ LLM
MODEL_ID = "TheBloke/Llama-2-7b-Chat-GGUF"
MODEL_BASENAME = "llama-2-7b-chat.Q4_K_M.gguf"
```

### 7. **Cách Theo Dõi Và Điều Chỉnh**
- Sử dụng lệnh `nvidia-smi` trong Command Prompt hoặc PowerShell để theo dõi mức sử dụng GPU.
- Nếu thấy GPU vượt quá giới hạn hoặc gặp lỗi "Out of Memory", hãy giảm `N_GPU_LAYERS` hoặc `N_BATCH`.

### Tóm Lại
Với GPU RTX 3060 (12GB VRAM), bạn có thể đặt `N_GPU_LAYERS` = 35, `N_BATCH` = 32, `CONTEXT_WINDOW_SIZE` = 1024, và `EMBEDDING_MODEL_NAME` là `hkunlp/instructor-large`. Cấu hình này sẽ giúp bạn xử lý hiệu quả tài liệu Bộ luật hình sự dài mà không gặp vấn đề về bộ nhớ hoặc hiệu suất.
