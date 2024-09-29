import os
import logging
import click
import torch
# import utils
from langchain.chains import RetrievalQA
# from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import \
    StreamingStdOutCallbackHandler  # dùng để hiển thị kết quả trả về theo từng dòng
from langchain.callbacks.manager import CallbackManager

# Quản lý callback cho việc hiển thị kết quả
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Import các hàm và lớp cần thiết từ các tệp module khác
from prompt_template_utils import get_prompt_template
from utils import get_embeddings
from langchain.vectorstores import Chroma
from transformers import (
    GenerationConfig,
)

# Import các hàm để tải mô hình khác nhau
from load_models import (
    load_quantized_model_awq,
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)

# Định nghĩa các hằng số (constants) sử dụng trong toàn bộ chương trình
from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    CHROMA_SETTINGS,
)

# Đảm bảo thư viện tạo pipeline đã được import trước khi chạy file
from transformers import pipeline

# Khởi tạo pipeline dịch với chỉ định `device` cho CUDA
translate_vi_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-vi-en", device=0)  # `device=0` chỉ định GPU
translate_en_to_vi = pipeline("translation", model="Helsinki-NLP/opus-mt-en-vi", device=0)


# Hàm dịch ngôn ngữ với điều kiện đầu vào tiếng Việt sang tiếng Anh và ngược lại
def translate(query, from_lang, to_lang):
    global translation
    print("Độ dài truy vấn gốc:", len(query))
    if from_lang == "vi" and to_lang == "en":
        if len(query) > 1000:
            query = query[:1000]  # Cắt chuỗi đầu vào nếu quá dài
        translation = translate_vi_to_en(query, max_length=1000)[0]['translation_text']
    elif from_lang == "en" and to_lang == "vi":
        if len(query) > 1000:
            query = query[:1000]
        translation = translate_en_to_vi(query, max_length=1000)[0]['translation_text']
    print("Độ dài truy vấn đã xử lý:", len(query))
    return translation


# Hàm load mô hình với thiết bị và định danh mô hình
def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
    """
    Chọn và tải mô hình text generation (sinh ngôn ngữ) sử dụng thư viện HuggingFace.
    Lần chạy đầu tiên, mô hình sẽ được tải từ HuggingFace. Các lần sau sẽ sử dụng từ ổ đĩa.

    Args:
        device_type (str): Loại thiết bị để chạy mô hình, ví dụ: "cuda" cho GPU hoặc "cpu" cho CPU.
        model_id (str): Định danh của mô hình được tải từ HuggingFace.
        model_basename (str, optional): Tên của mô hình nếu sử dụng mô hình đã được lượng tử hóa (quantized). Mặc định là None.

    Returns:
        HuggingFacePipeline: Đối tượng pipeline của mô hình sinh ngôn ngữ đã được tải.

    Raises:
        ValueError: Nếu mô hình hoặc loại thiết bị không được hỗ trợ.
    """
    logging.info(f"Tải Mô Hình: {model_id}, trên thiết bị: {device_type}")
    logging.info("Quá trình này có thể mất vài phút!")

    # Tải mô hình theo loại định dạng basename nếu được chỉ định
    if model_basename is not None:
        if ".gguf" in model_basename.lower():
            llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
            return llm
        elif ".ggml" in model_basename.lower():
            model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
        elif ".awq" in model_basename.lower():
            model, tokenizer = load_quantized_model_awq(model_id, LOGGING)
        else:
            model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
    else:
        model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

    # Tải cấu hình của mô hình để tránh các cảnh báo
    generation_config = GenerationConfig.from_pretrained(model_id)

    # Tạo một pipeline cho mô hình sinh ngôn ngữ
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_NEW_TOKENS,
        temperature=0.2,  # Điều chỉnh độ sáng tạo của mô hình
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Mô hình LLM cục bộ đã được tải thành công")

    return local_llm


# Hàm khởi tạo pipeline truy vấn và trả lời (retrieval-based QA)
def retrieval_qa_pipline(device_type, use_history, promptTemplate_type="llama"):
    """
    Khởi tạo và trả về pipeline truy vấn và trả lời dựa trên truy xuất thông tin (retrieval-based QA).

    Pipeline này sử dụng hệ thống nhúng ngữ nghĩa (embeddings) từ thư viện HuggingFace và truy xuất
    thông tin dựa trên các embedding đã được tính toán trước.

    Args:
        promptTemplate_type:
        device_type (str): Loại thiết bị để chạy mô hình, ví dụ: 'cpu', 'cuda', vv.
        use_history (bool): Quyết định xem có sử dụng lịch sử hội thoại hay không.

    Returns:
        RetrievalQA: Hệ thống QA dựa trên truy xuất thông tin đã được khởi tạo.

    Lưu ý:
    - Hàm này sử dụng các embeddings từ thư viện HuggingFace.
    - Sử dụng lớp `Chroma` để tải vector store chứa các embeddings đã tính toán.
    - Trình truy xuất (retriever) sẽ lấy thông tin liên quan dựa trên truy vấn đầu vào.
    - Tải mô hình LLM lên thiết bị được chỉ định bằng ID và basename.
    """

    # Tải embedding từ thiết bị đã chỉ định
    embeddings = get_embeddings(device_type)
    logging.info(f"Tải embeddings từ {EMBEDDING_MODEL_NAME}")

    # Tải vectorstore đã được lưu trữ
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()

    # Lấy mẫu prompt và bộ nhớ nếu được chỉ định
    prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)

    # Tải pipeline LLM
    llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)

    # Tạo hệ thống QA tùy theo việc sử dụng lịch sử hội thoại hay không
    if use_history:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    else:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            callbacks=callback_manager,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )

    return qa


# Định nghĩa các tùy chọn dòng lệnh cho việc chạy mô hình
@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice([
        "cpu", "cuda", "ipu", "xpu", "mkldnn", "opengl", "opencl", "ideep",
        "hip", "ve", "fpga", "ort", "xla", "lazy", "vulkan", "mps", "meta",
        "hpu", "mtia"
    ]),
    help="Thiết bị để chạy mô hình. (Mặc định là cuda nếu khả dụng)",
)
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Hiển thị nguồn tài liệu cùng với câu trả lời (Mặc định là False)",
)
@click.option(
    "--use_history",
    "-h",
    is_flag=True,
    help="Sử dụng lịch sử hội thoại (Mặc định là False)",
)
@click.option(
    "--model_type",
    default="llama",
    type=click.Choice(["llama", "mistral", "non_llama"]),
    help="Loại mô hình, llama, mistral hoặc non_llama",
)
@click.option(
    "--save_qa",
    is_flag=True,
    help="Lưu cặp câu hỏi và câu trả lời vào file CSV (Mặc định là False)",
)
def main(device_type, show_sources, use_history, model_type, save_qa):
    print(f"Đang chạy trên thiết bị: {device_type}")

    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    # Khởi tạo pipeline truy vấn và trả lời
    qa = retrieval_qa_pipline(device_type, use_history, promptTemplate_type=model_type)

    while True:
        query_vi = input("\nNhập câu hỏi (Tiếng Việt): ")
        if query_vi.lower() == "exit":
            break

        # Dịch câu hỏi từ Tiếng Việt sang Tiếng Anh
        query_en = translate(query_vi, "vi", "en")
        res = qa(query_en)
        answer_en = res["result"]
        answer_vi = translate(answer_en, "en", "vi")

        # In ra kết quả
        print("\n> Câu hỏi (Tiếng Việt):")
        print(query_vi)
        print("\n> Câu trả lời (Tiếng Việt):")
        print(answer_vi)

        if show_sources:
            print("----------------------------------TÀI LIỆU GỐC---------------------------")
            for document in res["source_documents"]:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print("----------------------------------TÀI LIỆU GỐC---------------------------")

        if save_qa:
            # Thêm hàm ghi câu hỏi và câu trả lời ra file CSV tại đây (nếu cần)
            pass


if __name__ == "__main__":
    main()
