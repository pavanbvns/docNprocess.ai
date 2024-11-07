import torch
import shutil
import magic
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from typing import List
from PIL import Image
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Image


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
            return loader.load()
        elif (
            file_path.endswith(".jpg")
            or file_path.endswith(".png")
            or file_path.endswith(".tiff")
        ):
            image = Image.open(file_path)
            return image
        # elif file_path.endswith(".doc") or file_path.endswith(".docx"):
        # text = docx2txt.process(file_path)
        # loader = TextLoader(text)
        else:
            print("Unsupported file type")

    def process_document(file_path, images_path):
        if file_path.endswith(".pdf"):
            raw_pdf_elements = partition_pdf(
                filename=file_path,
                extract_images_in_pdf=True,
                infer_table_structure=True,
                chunking_strategy="by_title",
                max_characters=4000,
                new_after_n_chars=3800,
                combine_text_under_n_chars=2000,
                image_output_dir_path=images_path,
            )

            tables_in_doc = []
            texts_in_doc = []

            # categorizing data based on tables and text
            for element in raw_pdf_elements:
                if "unstructured.documents.elements.Table" in str(type(element)):
                    tables_in_doc.append(str(element))
                elif "unstructured.documents.elements.CompositeElement" in str(
                    type(element)
                ):
                    texts_in_doc.append(str(element))

            images = [
                element for element in raw_pdf_elements if isinstance(element, Image)
            ]
            for i, img in enumerate(images):
                with open(f"{images_path}/image_{i}.png", "wb") as f:
                    f.write(img.data)

        return texts_in_doc, tables_in_doc

        def gen_table_desc(table_list, llama_text_gen_pipeline):
            # Configuration for text generation
            generation_args = {
                "max_new_tokens": 4000,
                "return_full_text": False,
                "temperature": 0.0,
                "do_sample": False,
            }

            # Prompt for generating descriptions
            prompt_text = """You are an assistant tasked with describing every row of the table. \
            Give a detailed description of every row of table. Table chunk: {element}. \
            Also give a summary of the table in the end."""

            # Generating descriptions for each table
            table_desc = []
            for text in table_list:
                output = llama_text_gen_pipeline(
                    [{"role": "user", "content": prompt_text.format(element=text)}],
                    **generation_args,
                )
                table_desc.append(output[0]["generated_text"])
            return table_desc
