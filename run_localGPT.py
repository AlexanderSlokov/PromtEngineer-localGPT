import os
import logging
import click
import torch
import utils
import nltk

# thư viện cho các chỉ số đánh giá

from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import precision_recall_fscore_support
from typing import List

from langchain.chains import RetrievalQA
# from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

from prompt_template_utils import get_prompt_template
from utils import get_embeddings

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import (
    GenerationConfig,
    pipeline,
)

from load_models import (
    load_quantized_model_awq,
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)

from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    CHROMA_SETTINGS,
)


def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        LOGGING:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

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

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.1,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm


def retrieval_qa_pipline(device_type, use_history, promptTemplate_type="llama"):
    """
    Initializes and returns a retrieval-based Question Answering (QA) pipeline.

    This function sets up a QA system that retrieves relevant information using embeddings
    from the HuggingFace library. It then answers questions based on the retrieved information.

    Parameters:
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'cuda', etc.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Returns:
    - RetrievalQA: An initialized retrieval-based QA system.

    Notes:
    - The function uses embeddings from the HuggingFace library, either instruction-based or regular.
    - The Chroma class is used to load a vector store containing pre-computed embeddings.
    - The retriever fetches relevant documents or data based on a query.
    - The prompt and memory, obtained from the `get_prompt_template` function, might be used in the QA system.
    - The model is loaded onto the specified device using its ID and basename.
    - The QA system retrieves relevant documents using the retriever and then answers questions based on those documents.
    """

    """
    (1) Chooses an appropriate langchain library based on the enbedding model name.  Matching code is contained within ingest.py.

    (2) Provides additional arguments for instructor and BGE models to improve results,
    pursuant to the instructions contained on
    their respective huggingface repository, project page or github repository.
    """

    embeddings = get_embeddings(device_type)

    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")

    # load the vectorstore
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()

    # get the prompt template and memory if set by the user.
    prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)

    # load the llm pipeline
    llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)

    if use_history:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    else:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )

    return qa


def calculate_metrics(predicted_answer: str, reference_answer: str, k_retrieved: List[str]) -> dict:
    """
    Tính toán các chỉ số đánh giá như BLEU, Recall@k, MRR và F1 Score.

    Args:
        predicted_answer (str): Câu trả lời được sinh ra bởi mô hình.
        reference_answer (str): Câu trả lời đúng hoặc câu trả lời chuẩn.
        k_retrieved (List[str]): Danh sách các tài liệu trả về (top k tài liệu).

    Returns:
        dict: Các chỉ số đánh giá được tính toán.
    """
    # Tính BLEU Score
    reference = [nltk.word_tokenize(reference_answer.lower())]  # Đưa về dạng từ viết thường
    candidate = nltk.word_tokenize(predicted_answer.lower())
    bleu_score = sentence_bleu(reference, candidate)

    # Tính Recall@k: Tỷ lệ tài liệu khớp với câu trả lời chuẩn trong top k tài liệu
    relevant_retrieved = sum([1 for doc in k_retrieved if reference_answer in doc])
    recall_at_k = relevant_retrieved / len(k_retrieved) if len(k_retrieved) > 0 else 0.0

    # Tính Mean Reciprocal Rank (MRR)
    mrr = 0.0
    for rank, doc in enumerate(k_retrieved, 1):
        if reference_answer in doc:
            mrr = 1 / rank
            break

    # Tính Precision, Recall và F1 Score
    precision, recall, f1, _ = precision_recall_fscore_support([reference_answer], [predicted_answer], average='macro')

    # Trả về các chỉ số
    return {
        'BLEU': bleu_score,
        'Recall@k': recall_at_k,
        'MRR': mrr,
        'F1 Score': f1
    }


def test_sample_query_offical(qa):
    """
    Hàm để kiểm tra hệ thống QA với câu hỏi mẫu và câu trả lời tham chiếu.

    Args:
        qa (RetrievalQA): Hệ thống QA đã được khởi tạo.
    """
    # Câu hỏi mẫu và câu trả lời tham chiếu bằng tiếng Việt
    query_vi = ("Điều 123 của Bộ luật Hình sự Việt Nam năm 2015 quy định như thế nào về tội giết người và các tình tiết "
                "tăng nặng liên quan đến tội này?")
    reference_answer = """
   Based on the provided context, I can provide information on the relevant laws and regulations in Vietnam related to murder and related offenses. According to Article 123 of the 2015 Criminal Code of Vietnam, whoever commits murder shall be punished with imprisonment from 15 years to life imprisonment or death penalty.
Additionally, according to Article 124 of the same code, whoever abets or assists in the commission of murder shall be punished with imprisonment from 10 years to 15 years or fine from VND 50 million to VND 100 million (approximately USD 2,200 to USD 4,400).
It is important to note that these provisions are subject to change and may have additional requirements or exceptions as specified in the law. Therefore, it is recommended to consult with legal professionals or seek advice from competent authorities for further clarification.

    """

    # Gọi hệ thống QA trực tiếp với câu hỏi tiếng Việt
    res = qa(query_vi)
    answer_vi = res["result"]

    # In ra kết quả
    print("\n> Câu hỏi (Tiếng Việt):")
    print(query_vi)
    print("\n> Câu trả lời (Tiếng Việt):")
    print(answer_vi)

    # Gọi hàm đánh giá với câu trả lời đã sinh ra từ mô hình
    benchmark_metrics = calculate_metrics(predicted_answer=answer_vi, reference_answer=reference_answer,
                                          k_retrieved=[doc.page_content for doc in res["source_documents"]])

    print(
        f"\n> Các chỉ số đánh giá:\nBLEU: {benchmark_metrics['BLEU']}\nRecall@k: {benchmark_metrics['Recall@k']}\nMRR: "
        f"{benchmark_metrics['MRR']}\nF1 Score: {benchmark_metrics['F1 Score']}")


# chose device typ to run on as well as to show source documents.
@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)
@click.option(
    "--use_history",
    "-h",
    is_flag=True,
    help="Use history (Default is False)",
)
@click.option(
    "--model_type",
    default="llama",
    type=click.Choice(
        ["llama", "mistral", "non_llama"],
    ),
    help="model type, llama, mistral or non_llama",
)
@click.option(
    "--save_qa",
    is_flag=True,
    help="whether to save Q&A pairs to a CSV file (Default is False)",
)
def main(device_type, show_sources, use_history, model_type, save_qa):
    """
    Implements the main information retrieval task for a localGPT.

    This function sets up the QA system by loading the necessary embeddings, vectorstore, and LLM model.
    It then enters an interactive loop where the user can input queries and receive answers. Optionally,
    the source documents used to derive the answers can also be displayed.

    Parameters:
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'mps', 'cuda', etc.
    - show_sources (bool): Flag to determine whether to display the source documents used for answering.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Notes:
    - Logging information includes the device type, whether source documents are displayed, and the use of history.
    - If the models directory does not exist, it creates a new one to store models.
    - The user can exit the interactive loop by entering "exit".
    - The source documents are displayed if the show_sources flag is set to True.

    """

    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")
    logging.info(f"Use history set to: {use_history}")

    # check if models directory do not exist, create a new one and store models here.
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    qa = retrieval_qa_pipline(device_type, use_history, promptTemplate_type=model_type)
    # Interactive questions and answers

    # test_sample_query_offical(qa)

    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        # Get the answer from the chain
        res = qa(query)
        answer, docs = res["result"], res["source_documents"]

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        if show_sources:  # this is a flag that you can set to disable showing answers.
            # # Print the relevant sources used for the answer
            print("----------------------------------SOURCE DOCUMENTS---------------------------")
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print("----------------------------------SOURCE DOCUMENTS---------------------------")

        # Log the Q&A to CSV only if save_qa is True
        if save_qa:
            utils.log_to_csv(query, answer)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
