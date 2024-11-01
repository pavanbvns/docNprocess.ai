import torch
import shutil
import magic
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from typing import List


class utils:
    def get_device_map() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def pdf_loader(file_location: str):
        loader = PyPDFDirectoryLoader(file_location)
        docs = loader.load()
        return docs

    def text_splitter(docs):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=64
        )
        texts = text_splitter.split_documents(docs)
        return texts

    def get_doc_mime_type(doc):
        doc_mime_type = magic.from_file(doc)
        return doc_mime_type

    def upload_file(uploaded_file):
        path = f"uploads/{uploaded_file.filename}"
        with open(path, "w+b") as file:
            shutil.copyfileobj(uploaded_file.file, file)

        return {
            "file": uploaded_file.filename,
            "content": uploaded_file.content_type,
            "path": path,
        }

    def load_document(file_path: str) -> List[str]:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            """  elif (
                file_path.endswith(".jpg")
                or file_path.endswith(".png")
                or file_path.endswith(".tiff")
            ):
                image = Image.open(file_path)
                text = pytesseract.image_to_string(image)
                loader = TextLoader(text)
            elif file_path.endswith(".doc") or file_path.endswith(".docx"):
                text = docx2txt.process(file_path)
                loader = TextLoader(text)"""
        else:
            print("Unsupported file type")
        return loader.load()
