from transformers import AutoTokenizer

# Sử dụng tokenizer của mô hình để tính toán số lượng tokens
tokenizer = AutoTokenizer.from_pretrained("hkunlp/instructor-large")

# Giả sử bạn đã có một điều luật dưới dạng chuỗi (string)
sample_text = "Điều 1. Phạm vi điều chỉnh. Bộ luật này quy định những nguyên tắc cơ bản của pháp luật hình sự, hình phạt và các biện pháp tư pháp khác đối với người phạm tội..."

# Tính số lượng tokens của điều luật
tokens = tokenizer.encode(sample_text)
print(f"Số lượng tokens: {len(tokens)}")
