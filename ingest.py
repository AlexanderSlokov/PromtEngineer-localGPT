import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
import click
import torch
from langchain.docstore.document import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from utils import get_embeddings

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)


def file_log(logentry):
    with open("file_ingest.log", "a") as file1:
        file1.write(logentry + "\n")
    logging.info(logentry)


def load_single_document(file_path: str) -> Document:
    try:
        file_extension = os.path.splitext(file_path)[1]
        logging.info(f"Attempting to load file with extension: {file_extension}")
        loader_class = DOCUMENT_MAP.get(file_extension)
        if loader_class:
            logging.info(f"{file_path} loader found.")
            loader = loader_class(file_path)
        else:
            logging.error(f"{file_path} document type is undefined.")
            raise ValueError("Document type is undefined")
        document = loader.load()[0]
        logging.info(f"{file_path} loaded successfully.")
        return document
    except Exception as ex:
        logging.error(f"{file_path} loading error: {ex}")


def load_document_batch(filepaths):
    logging.info("Loading document batch")
    with ThreadPoolExecutor(max_workers=min(len(filepaths), INGEST_THREADS)) as executor:
        futures = [executor.submit(load_single_document, filepath) for filepath in filepaths]
        if not futures:
            file_log("No files to submit")
            return None
        else:
            data_list = [future.result() for future in futures if future.result() is not None]
            return data_list, filepaths


def load_documents(source_dir: str) -> list[Document]:
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            logging.info(f"Importing: {file_name}")
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in DOCUMENT_MAP.keys():
                paths.append(source_file_path)

    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        for i in range(0, len(paths), chunksize):
            filepaths = paths[i: (i + chunksize)]
            try:
                future = executor.submit(load_document_batch, filepaths)
            except Exception as ex:
                file_log(f"executor task failed: {ex}")
                future = None
            if future is not None:
                futures.append(future)
        for future in as_completed(futures):
            try:
                contents, _ = future.result()
                docs.extend(contents)
            except Exception as ex:
                file_log(f"Exception: {ex}")

    return docs


def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    text_docs, python_docs = [], []
    for doc in documents:
        if doc is not None:
            file_extension = os.path.splitext(doc.metadata["source"])[1]
            if file_extension == ".py":
                python_docs.append(doc)
            else:
                text_docs.append(doc)
    return text_docs, python_docs


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
def main(device_type):
    # Load documents and split in chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    if not documents:
        logging.error("No documents loaded. Check the source directory and file formats.")
        return  # Exit if no documents are loaded

    text_documents, python_documents = split_documents(documents)
    if not text_documents and not python_documents:
        logging.error("Document split failed or resulted in no usable documents.")
        return  # Exit if splitting documents resulted in empty lists

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=880, chunk_overlap=200
    )

    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))

    if not texts:
        logging.error("No text chunks were created. Check document content and splitter configuration.")
        return  # Exit if no text chunks were created

    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    logging.info(f"Embedding model being used: {EMBEDDING_MODEL_NAME}")
    embeddings = get_embeddings(device_type)
    if not embeddings:
        logging.error("Failed to load embeddings. Check the embedding model and device type.")
        return  # Exit if embeddings couldn't be loaded

    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")

    try:
        db = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=PERSIST_DIRECTORY,
            client_settings=CHROMA_SETTINGS,
        )
        logging.info("Database creation successful.")
    except Exception as e:
        logging.error(f"Failed to create Chroma database: {e}")
        return  # Exit if creating the database fails

    logging.info("Chroma database created successfully with the specified embeddings and documents.")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
