from fastapi import (
    FastAPI,
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
    Request,
    Query,
)
import uvicorn
import os
import shutil
import uuid
import gc
import torch
from typing import List, Optional
from PIL import Image
from utils.utils import Utils, JobTracker
from utils.model import ModelLoader
from utils.vectors import VectorStore
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.doc import partition_docx
from config_loader import load_config, setup_logging, display_banner
import traceback
import threading
from pdf2image import convert_from_path
from docx import Document


print("Running initial garbage collection...")
gc.collect()

if torch.cuda.is_available():
    print("Clearing GPU memory...")
    torch.cuda.empty_cache()

torch.set_num_threads(8)


def create_app():
    # Set environment variable for CUDA memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    uvicorn_url = os.getenv("UVICORN_URL", "http://127.0.0.1:8000")
    qdrant_server_url = os.getenv("QDRANT_SERVER_URL", "http://127.0.0.1:6333")

    config = load_config("config.yml")
    logger = setup_logging(config["directories"]["log_file_path"])

    # Load configuration and initialize FastAPI app
    logger.info("Initializing FastAPI app, router and job tracker...")
    app = FastAPI()
    router = APIRouter()
    job_tracker = JobTracker()

    logger.info("Loading LLM")
    # Determine which models to load based on config

    # running GC again
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Load models based on pipeline configuration
    model_loader = ModelLoader(config_file_path="config.yml")

    logger.info("LLM's are now loaded and initialized")
    # Initialize vector store with appropriate models
    logger.info("Initializing vector store with appropriate models")
    vector_store = VectorStore(
        model_loader=model_loader,
        tokenizer=model_loader.tokenizer,
        processor=model_loader.processor,
        config_file="config.yml",
    )

    # Ensure necessary directories
    logger.info("Final checks of app initialization including directory checks")
    UPLOAD_DIR = config["directories"]["upload_dir"]
    PROCESSED_DIR = config["directories"]["processed_dir"]
    Utils.ensure_directory_exists(UPLOAD_DIR)
    Utils.ensure_directory_exists(PROCESSED_DIR)

    logger.info("App is now ready for usage.")

    @router.get("/")
    async def home():
        return {"message": "Doc.ai service by Pavan Kumar B V N S"}

    @app.post("/get_file_summary")
    # @profile
    def get_file_summary(
        request: Request,
        uploaded_file: UploadFile = File(...),
        min_words: int = 50,
        max_words: int = 200,
        summarization_depth: str = "simple",
    ):
        # Starting a new job
        logger.info(f"Starting a new file summarization ({summarization_depth}) job")
        job_id = job_tracker.create_job()

        # 2. destination local directory to which file has to be downloaded
        file_path = f"{UPLOAD_DIR}/{uploaded_file.filename}"

        if summarization_depth not in ["simple", "thorough"]:
            raise HTTPException(status_code=400, detail="Invalid summarization depth.")
        if min_words > max_words:
            raise HTTPException(
                status_code=400, detail="min_words cannot exceed max_words."
            )

        #  validating file size
        if Utils.file_size_within_limit(
            uploaded_file, config["settings"]["max_file_size_mb"]
        ) and Utils.validate_file_type(
            uploaded_file, config["settings"]["allowed_file_extensions"]
        ):
            uploaded_file.file.seek(0, 0)
            # saving the file
            with open(file_path, "wb") as file:
                shutil.copyfileobj(uploaded_file.file, file)
        else:
            raise HTTPException(
                status_code=400, detail="File exceeds size limit or unsupported type."
            )
        file_extension = uploaded_file.filename.split(".")[-1].lower()

        print("file_extension is :", file_extension)

        if file_extension in ("pdf", "doc", "docx"):
            page_count = count_pages_in_document(file_path, file_extension)
            if page_count <= 5:
                # Convert pages to images
                images_dir = os.path.join(PROCESSED_DIR, f"{job_id}_images")
                images = convert_document_pages_to_images(
                    file_path, file_extension, images_dir
                )

                # Generate summary using the Vision model
                summaries = []
                for image_path in images:
                    image = Image.open(image_path).convert("RGB")
                    summary = model_loader.generate_image_summary(
                        image, summarization_depth, min_words, max_words
                    )
                    summaries.append(summary)

                final_summary = " ".join(summaries)
                logger.info(f"Generated image-based summary: {final_summary}")
                job_tracker.update_job_status(job_id, "Completed")
                print(
                    "------------------------Summary----------------------------------- \n",
                    summary,
                )
                return {"job_id": job_id, "summary": summary}

            else:
                # Extract content and create embeddings
                logger.info("initiating content extraction")
                file_extension = uploaded_file.filename.split(".")[-1].lower()
                try:
                    text_elements, table_elements, image_elements = (
                        extract_content_elements(file_path, file_extension, request)
                    )
                    logger.info("Content extraction of uploaded file is complete")
                except Exception as e:
                    logger.error(f"Something went wrong (job_id: {job_id}): {e}")
                    raise HTTPException(
                        status_code=500, detail="Content extraction failed."
                    )

                if summarization_depth == "simple":
                    try:
                        logger.info("Initiating simple - text summarization task")
                        summary = generate_simple_summary(
                            text_elements, table_elements, min_words, max_words, request
                        )
                    except Exception as e:
                        job_tracker.update_job_status(job_id, "Aborted")
                        logger.error(
                            f"Error in simple - text summarization taskmary (job_id: {job_id}): {e}"
                        )
                        raise HTTPException(
                            status_code=500, detail="Failed to generate summary."
                        )
                    logger.info("simple - text summarization task is now complete")
                else:
                    try:
                        logger.info("Initiating thorough - text summarization task")
                        summary = generate_thorough_summary(
                            text_elements,
                            table_elements,
                            image_elements,
                            min_words,
                            max_words,
                            request,
                        )
                    except Exception as e:
                        job_tracker.update_job_status(job_id, "Aborted")
                        logger.error(
                            f"Error in generate_file_summary (job_id: {job_id}): {e}"
                        )
                        raise HTTPException(
                            status_code=500, detail="Failed to generate summary."
                        )
                    logger.info("Thorough - text summarization task is now complete")
        elif file_extension in ("png", "jpg", "tiff", "jpeg"):
            image = Image.open(file_path).convert("RGB")
            summary = model_loader.generate_image_summary(
                image, summarization_depth, min_words, max_words
            )
            return summary
        logger.info("Initiating post job cleanup")
        try:
            processed_file_path = os.path.join(PROCESSED_DIR, file.filename)
            logger.info("Moving the processed files")
            shutil.move(file_path, processed_file_path)
            if os.path.exists(file_path):
                os.remove(file_path)
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logger.info("Error in post task cleanup (job_id: {job_id}): {e}")
        job_tracker.update_job_status(job_id, "Completed")
        print(
            "------------------------Summary----------------------------------- \n",
            summary,
        )
        return {"job_id": job_id, "summary": summary}

    @app.post("/qna_on_file")
    def qna_on_file(
        request: Request,
        uploaded_file: UploadFile = File(...),
        questions: List[str] = Query(None),
        answer_type: str = "specific",
    ):
        """
        Processes uploaded documents for question answering.
        Handles both image and document files intelligently and robustly.
        """
        logger.info("Starting a new Q&A task on the uploaded document.")
        job_id = job_tracker.create_job()
        responses = []  # Store responses for all questions
        combined_summary = None  # Placeholder for combined summary (if needed)
        print(questions)
        try:
            # Save the file locally
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.filename)
            with open(file_path, "wb") as file:
                shutil.copyfileobj(uploaded_file.file, file)

            # Validate file type and extract extension
            file_extension = uploaded_file.filename.split(".")[-1].lower()
            if not Utils.validate_file_type(
                uploaded_file, config["settings"]["allowed_file_extensions"]
            ):
                raise HTTPException(status_code=400, detail="Unsupported file type.")

            # Handle document-based Q&A (PDF, DOC, DOCX)
            if file_extension in ("pdf", "doc", "docx"):
                page_count = count_pages_in_document(file_path, file_extension)

                if page_count <= 3:
                    # Convert document to images for processing
                    logger.info(
                        "Document has less than or equal to 3 pages. Converting to images..."
                    )
                    images_dir = os.path.join(PROCESSED_DIR, f"{job_id}_images")
                    images = convert_document_pages_to_images(
                        file_path, file_extension, images_dir
                    )

                    # Generate summaries for all images
                    combined_summary = ""
                    for image_path in images:
                        image = Image.open(image_path)
                        summary = model_loader.generate_image_summary(
                            image, "thorough", min_words=50, max_words=200
                        )
                        combined_summary += summary + " "

                    logger.info(
                        f"Combined summary for all images generated.\n{combined_summary}"
                    )

                    # Use the combined summary to answer questions
                    for question in questions:
                        try:
                            answer = model_loader.answer_question_with_text(
                                combined_summary, question, answer_type
                            )
                            responses.append({"question": question, "answer": answer})
                        except Exception as e:
                            logger.error(
                                f"Error answering question '{question}': {e}. Skipping."
                            )
                            responses.append(
                                {
                                    "question": question,
                                    "answer": "Failed to generate an answer.",
                                }
                            )

                else:
                    # Handle larger documents by extracting content
                    logger.info("Extracting text, tables, and images from document...")
                    try:
                        # text_elements, table_elements, image_elements = (
                        #     extract_content_elements(file_path, file_extension, request)
                        # )
                        elements_with_embeddings = extract_content_covert_to_embeddings(
                            file_path,
                            file_extension,
                        )
                        logger.info("Content extraction completed.")
                    except Exception as e:
                        logger.error(f"Content extraction failed: {e}")
                        job_tracker.update_job_status(job_id, "Aborted")
                        raise HTTPException(
                            status_code=500, detail="Content extraction failed."
                        )

                    # Upsert embeddings for content elements
                    try:
                        vector_store.upsert_embedding(
                            elements_with_embeddings,
                            file_path,
                            job_id,
                        )
                        logger.info("Content embeddings upserted successfully.")
                    except Exception as e:
                        logger.error(f"Upsert embeddings failed: {e}")
                        job_tracker.update_job_status(job_id, "Aborted")
                        raise HTTPException(
                            status_code=500, detail="Failed to upsert embeddings."
                        )

                    # Query embeddings for Q&A
                    try:
                        responses = vector_store.query_embeddings_on_file(
                            file_path, questions, top_k=1
                        )
                        logger.info(
                            f"Q&A task completed with {len(responses)} responses."
                        )
                    except Exception as e:
                        logger.error(f"Error querying embeddings: {e}")
                        job_tracker.update_job_status(job_id, "Aborted")
                        raise HTTPException(
                            status_code=500,
                            detail="Error in question answering for one or more questions.",
                        )

            # Handle image-based Q&A (PNG, JPG, TIFF, JPEG)
            elif file_extension in ("png", "jpg", "tiff", "jpeg"):
                try:
                    # Open and preprocess the image
                    image = Image.open(file_path)
                    if not isinstance(image, Image.Image):
                        raise ValueError(
                            "Failed to load the image as a valid PIL.Image."
                        )
                    for question in questions:
                        try:
                            answer = model_loader.answer_question_with_image(
                                image, question, answer_type
                            )
                            responses.append({"question": question, "answer": answer})
                        except Exception as e:
                            logger.error(f"Error answering question '{question}': {e}")
                            responses.append({"question": question, "answer": "Error"})

                except Exception as e:
                    logger.error(
                        f"Error during image-based Q&A task (job_id: {job_id}): {str(e)}"
                    )
                    job_tracker.update_job_status(job_id, "Aborted")
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to process image-based question answering.",
                    )

            else:
                raise HTTPException(status_code=400, detail="Unsupported file type.")

            # Cleanup and finalize job
            logger.info("Initiating post-job cleanup.")
            processed_file_path = os.path.join(PROCESSED_DIR, uploaded_file.filename)
            shutil.move(file_path, processed_file_path)
            if os.path.exists(file_path):
                os.remove(file_path)
            torch.cuda.empty_cache()
            gc.collect()
            job_tracker.update_job_status(job_id, "Completed")

            # Finalize response
            if combined_summary:
                return {
                    "job_id": job_id,
                    "summary": combined_summary,
                    "responses": responses,
                }
            return {"job_id": job_id, "responses": responses}

        except Exception as e:
            job_tracker.update_job_status(job_id, "Aborted")
            logger.error(f"Error in Q&A task (job_id: {job_id}): {e}")
            raise HTTPException(status_code=500, detail="Failed to process Q&A.")

    @app.get("/jobs")
    async def get_job_history(request: Request):
        """Retrieves job history."""
        try:
            return job_tracker.get_job_history()
        except Exception as e:
            logger.error(f"Error in get_job_history: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve job history."
            )

    # ---------------------------Helper functions--------------------------------

    def generate_simple_summary(
        text_elements: List[str],
        table_elements: List[str],
        min_words: int,
        max_words: int,
        request: Request,
    ) -> str:
        combined_text = " ".join(text_elements + table_elements)
        # summary_prompt = f"Summarize the following content in between {min_words} and {max_words} words:\n{combined_text}"
        summary_prompt = f"""# IDENTITY and PURPOSE

            You are an expert content summarizer. You take content in and output a Markdown formatted summary using the format below.

            Take a deep breath and think step by step about how to best accomplish this goal using the following steps.

            # OUTPUT SECTIONS

            - Combine all of your understanding of the content into a single section, with {min_words} and {max_words} words in a section called OVERALL SUMMARY:.

            - Output the 10 most important points of the content as a list with no more than 15 words per point into a section called MAIN POINTS:.

            - Output a list of the 5 best takeaways from the content in a section called TAKEAWAYS:.

            # OUTPUT INSTRUCTIONS

            - Create the output using the formatting above.
            - You only output human readable Markdown.
            - Output numbered lists, not bullets.
            - Do not output warnings or notes—just the requested sections.
            - Do not repeat items in the output sections.
            - Do not start items with the same opening words.

            # INPUT:

            INPUT: \n{combined_text}"""
        summary = model_loader.generate_text(summary_prompt, max_length=max_words)
        return summary

    def generate_thorough_summary(
        text_elements: List[str],
        table_elements: List[str],
        image_elements: List[Image.Image],
        min_words: int,
        max_words: int,
        request: Request,
    ) -> str:
        table_summary = []
        image_summary = []

        def summarize_table(table, request: Request):
            prompt = f"Summarize the following table data:\n{table}"
            summary = model_loader.generate_text(prompt, max_length=max_words)
            table_summary.append(summary)

        def summarize_image(image, request: Request):
            image_text = model_loader.process_image(image)
            image_summary.append(f"Image summary:\n{image_text}")

        # Use threading for parallel table and image summarization
        threads = []
        for table in table_elements:
            thread = threading.Thread(target=summarize_table, args=(table,))
            threads.append(thread)
            thread.start()

        for image in image_elements:
            thread = threading.Thread(target=summarize_image, args=(image,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        combined_summary = (
            f"Summarize the following document with {min_words} to {max_words} words, "
            f"including tables and images summaries:\nText:\n{' '.join(text_elements)}\n"
            f"Tables:\n{' '.join(table_summary)}\nImages:\n{' '.join(image_summary)}"
        )
        summary = model_loader.model.generate_text(
            combined_summary, max_length=max_words
        )
        return summary

    def extract_content_elements(file_path, file_type, request: Request):
        text_elements, table_elements, image_elements = [], [], []
        if file_type in ["pdf", "docx"]:
            if file_type == "pdf":
                content = partition_pdf(
                    filename=file_path,
                    extract_images_in_pdf=True,
                    infer_table_structure=True,
                    chunking_strategy="by_title",
                )
            elif file_type == "docx":
                content = partition_docx(filename=file_path)

            for element in content:
                # Check if element has text content
                if hasattr(element, "text") and element.text:
                    text_elements.append(element.text)

                # Check if element can be interpreted as a table or contains tabular data
                elif hasattr(element, "category") and element.category == "Table":
                    # Attempt to get table content or representation as text
                    table_text = (
                        element.text if hasattr(element, "text") else str(element)
                    )
                    table_elements.append(table_text)

                # Check if element can be processed as an image
                elif hasattr(element, "image"):
                    try:
                        image = Image.open(element.image).convert("RGB")
                        image_elements.append(image)
                    except Exception as e:
                        logger.error(f"Failed to process image element: {e}")
        else:
            raise HTTPException(
                status_code=400, detail="Unsupported file type for extraction"
            )
        return text_elements, table_elements, image_elements

    def extract_content_covert_to_embeddings(file_path: str, file_type: str):
        """Extracts text, tables, and images from a document based on file type while preserving sequence."""
        elements_with_embeddings = []

        if file_type in ["pdf", "docx"]:
            if file_type == "pdf":
                content = partition_pdf(
                    filename=file_path,
                    extract_images_in_pdf=True,
                    infer_table_structure=True,
                    chunking_strategy="by_title",
                )
            elif file_type == "docx":
                content = partition_docx(filename=file_path)

            sequence_number = 0  # To track the order of elements
            for element in content:
                if hasattr(element, "text") and element.text:
                    embedding = model_loader.get_text_embeddings(element.text)
                    elements_with_embeddings.append(
                        {
                            "sequence_number": sequence_number,
                            "type": "text",
                            "content": element.text,
                            "embedding": embedding,
                        }
                    )
                    sequence_number += 1

                if hasattr(element, "category") and element.category == "Table":
                    table_text = (
                        element.text if hasattr(element, "text") else str(element)
                    )
                    embedding = model_loader.get_text_embeddings(table_text)
                    elements_with_embeddings.append(
                        {
                            "sequence_number": sequence_number,
                            "type": "table",
                            "content": table_text,
                            "embedding": embedding,
                        }
                    )
                    sequence_number += 1

                if hasattr(element, "image"):
                    image = Image.open(element.image).convert("RGB")
                    embedding = model_loader.get_image_embeddings(image)
                    image_summary = model_loader.generate_image_summary(
                        image, summarization_depth="simple", min_words=50, max_words=500
                    )
                    elements_with_embeddings.append(
                        {
                            "sequence_number": sequence_number,
                            "type": "image",
                            "content": image_summary,
                            "embedding": embedding,
                        }
                    )
                    sequence_number += 1

        else:
            # If file is a standalone image
            image = Image.open(file_path).convert("RGB")
            embedding = model_loader.get_image_embeddings(image)
            elements_with_embeddings = [
                {
                    "sequence_number": 0,
                    "type": "image",
                    "content": "Standalone Image Content",
                    "embedding": embedding,
                }
            ]

        return elements_with_embeddings

    def convert_doc_to_pdf(doc_path, output_path):
        """Converts a DOC/DOCX file to PDF using PyPDF."""
        try:
            import comtypes.client

            word = comtypes.client.CreateObject("Word.Application")
            doc = word.Documents.Open(doc_path)
            doc.SaveAs(output_path, FileFormat=17)  # 17 corresponds to wdFormatPDF
            doc.Close()
            word.Quit()
        except Exception as e:
            raise RuntimeError(f"Failed to convert DOC/DOCX to PDF: {e}")

    def count_pages_in_document(file_path, file_type):
        """Counts the number of pages in a PDF or DOC/DOCX document."""
        try:
            if file_type == "pdf":
                from PyPDF2 import PdfReader

                reader = PdfReader(file_path)
                return len(reader.pages)
            elif file_type in ["doc", "docx"]:
                doc = Document(file_path)
                return len(
                    doc.paragraphs
                )  # Approximation, better converted to PDF for accurate page count
            else:
                raise ValueError("Unsupported file type for page counting.")
        except Exception as e:
            raise RuntimeError(
                f"Error while counting pages in {file_type} document: {e}"
            )

    def convert_document_pages_to_images(file_path, file_type, output_dir):
        """Converts document pages to images if the page count is less than 5."""
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Convert to images
            if file_type == "pdf":
                images = convert_from_path(
                    file_path,
                    fmt="jpeg",
                    output_folder=output_dir,
                )
                image_paths = []

                # Save images with consistent naming
                for i, image in enumerate(images):
                    image_filename = os.path.join(
                        output_dir, f"{i}.jpeg"
                    )  # Named sequentially
                    image.save(image_filename, "JPEG")
                    image_paths.append(image_filename)
            elif file_type in ["doc", "docx"]:
                # Convert DOC/DOCX to PDF first
                temp_pdf_path = os.path.join(output_dir, "temp.pdf")
                convert_doc_to_pdf(file_path, temp_pdf_path)
                images = convert_from_path(
                    temp_pdf_path,
                    fmt="jpeg",
                    output_folder=output_dir,
                )
                os.remove(temp_pdf_path)

                image_paths = []

                # Save images with consistent naming
                for i, image in enumerate(images):
                    image_filename = os.path.join(
                        output_dir, f"{i}.jpeg"
                    )  # Named sequentially
                    image.save(image_filename, "JPEG")
                    image_paths.append(image_filename)
            else:
                raise ValueError("Unsupported file type for image conversion.")
            return image_paths
        except Exception as e:
            raise RuntimeError(f"Error while converting document pages to images: {e}")

    app.include_router(router)

    return app


# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    display_banner()
    app = create_app()
    uvicorn.run(app, reload=False, port=6060, host="0.0.0.0")

#  to be used later

#


# @app.post("/create_embeddings")
# async def create_embeddings(
#     background_tasks: BackgroundTasks,
#     files: List[UploadFile] = File(...),
#     collection_name: Optional[str] = None,
# ):
#     """Creates and stores embeddings for new files in a specified collection asynchronously."""
#     job_id = job_tracker.create_job()

#     def background_embedding_creation(file_paths):
#         """Background task to create embeddings and store in the vector store."""
#         try:
#             # Create embeddings for each file
#             vector_store.create_embeddings(
#                 file_paths, collection_name or vector_store.collection_name
#             )
#             job_tracker.update_job_status(job_id, "Completed")
#         except Exception as e:
#             job_tracker.update_job_status(job_id, "Aborted")
#             logger.error(
#                 f"Error in background embedding creation (job_id: {job_id}): {e}"
#             )
#             raise HTTPException(
#                 status_code=500, detail="Failed to create embeddings."
#             )
#         finally:
#             for file_path in file_paths:
#                 if os.path.exists(file_path):
#                     os.remove(file_path)
#             torch.cuda.empty_cache()
#             gc.collect()

#     try:
#         # Check cumulative file size
#         cumulative_size = sum(file.spool_max_size for file in files)
#         if cumulative_size > config["settings"]["max_cumulative_file_size_mb"]:
#             raise HTTPException(
#                 status_code=400, detail="Cumulative file size exceeds limit."
#             )

#         file_paths = []
#         for file in files:
#             file_path = os.path.join(UPLOAD_DIR, file.filename)
#             with open(file_path, "wb") as buffer:
#                 shutil.copyfileobj(file.file, buffer)
#             file_paths.append(file_path)

#         # Add the embedding creation task to background tasks
#         background_tasks.add_task(background_embedding_creation, file_paths)

#         job_tracker.update_job_status(job_id, "In Progress")
#         return {
#             "job_id": job_id,
#             "status": "Embedding creation started in the background.",
#         }

#     except Exception as e:
#         job_tracker.update_job_status(job_id, "Aborted")
#         logger.error(f"Error in create_embeddings (job_id: {job_id}): {e}")
#         raise HTTPException(status_code=500, detail="Failed to create embeddings.")

#     finally:
#         torch.cuda.empty_cache()
#         gc.collect()

# @app.get("/query_vector_db")
# async def query_vector_db(
#     query_text: Optional[str] = None,
#     query_image: Optional[UploadFile] = None,
#     top_k: int = 5,
# ):
#     """Queries the vector database for similar embeddings."""
#     job_id = job_tracker.create_job()
#     try:
#         if query_text:
#             result = vector_store.query_embeddings(
#                 query_text=query_text, top_k=top_k
#             )
#         elif query_image:
#             image_path = os.path.join(UPLOAD_DIR, query_image.filename)
#             with open(image_path, "wb") as buffer:
#                 shutil.copyfileobj(query_image.file, buffer)
#             result = vector_store.query_embeddings(
#                 query_image=image_path, top_k=top_k
#             )
#             os.remove(image_path)
#         else:
#             raise HTTPException(
#                 status_code=400, detail="Provide query_text or query_image."
#             )

#         job_tracker.update_job_status(job_id, "Completed")
#         return {"job_id": job_id, "result": result}

#     except Exception as e:
#         job_tracker.update_job_status(job_id, "Aborted")
#         logger.error(f"Error in query_vector_db (job_id: {job_id}): {e}")
#         raise HTTPException(
#             status_code=500, detail="Failed to query vector database."
#         )

#     finally:
#         torch.cuda.empty_cache()
#         gc.collect()


# @app.post("/qna_on_docs")
# def question_answering(
#     request: Request,
#     files: List[UploadFile] = File(...),
#     questions: List[str] = [],
#     answer_type: str = "specific",
# ):
#     """Processes uploaded documents for question answering."""
#     logger.info("Starting a new Q & A task on the uploaded documents")
#     job_id = job_tracker.create_job()
#     try:
#         # Check cumulative file size
#         cumulative_size = sum(
#             file.file.seek(0, 2)
#             or file.file.tell() / (1024 * 1024)
#             or file.file.seek(0)
#             for file in files
#         )
#         if cumulative_size > config["settings"]["max_cumulative_file_size_mb"]:
#             job_tracker.update_job_status(job_id, "Aborted")
#             raise HTTPException(
#                 status_code=400, detail="Cumulative file size exceeds limit."
#             )
#         if not all(
#             Utils.validate_file_type(
#                 file.filename, config["settings"]["allowed_file_extensions"]
#             )
#             for file in files
#         ):
#             job_tracker.update_job_status(job_id, "Aborted")
#             raise HTTPException(
#                 status_code=400,
#                 detail="One or more uploaded files are not supported by the application for this task.",
#             )
#         logger.info("Document size and file type validation complete")
#         responses = []
#         logger.info(
#             "Extracting content, convert the content into embeddings and upserting the embeddings to vectore db"
#         )
#         for file in files:
#             file_path = os.path.join(UPLOAD_DIR, file.filename)
#             with open(file_path, "wb") as buffer:
#                 shutil.copyfileobj(file.file, buffer)

#             # Process each file
#             file_extension = file.filename.split(".")[-1].lower()
#             file_uuid = str(uuid.uuid4())
#             file_hash = Utils.calculate_file_hash(file_path)

#             if not vector_store._check_existing_embedding(file_hash):
#                 try:
#                     # Extract content and create embeddings
#                     text_embeddings, table_embeddings, image_embeddings = (
#                         extract_content_covert_to_embeddings(
#                             file_path, file_extension
#                         )
#                     )
#                 except Exception as e:
#                     job_tracker.update_job_status(job_id, "Aborted")
#                     logger.error(
#                         f"Error in extract_content_covert_to_embeddings (job_id: {job_id}) is now aborted: {e}"
#                     )
#                     raise HTTPException(
#                         status_code=500,
#                         detail="Error in extract_content_covert_to_embeddings",
#                     )
#                 try:
#                     for embedding in text_embeddings:
#                         vector_store.upsert_embedding(
#                             embedding, "text", file_hash, file_uuid
#                         )
#                     for embedding in table_embeddings:
#                         vector_store.upsert_embedding(
#                             embedding, "table", file_hash, file_uuid
#                         )
#                     for embedding in image_embeddings:
#                         vector_store.upsert_embedding(
#                             embedding, "image", file_hash, file_uuid
#                         )
#                 except Exception as e:
#                     job_tracker.update_job_status(job_id, "Aborted")
#                     logger.error(
#                         f"Error in upserting embedding to Vector DB (job_id: {job_id}): {e}"
#                     )
#                     raise HTTPException(
#                         status_code=500,
#                         detail="Error in upserting embedding to Vector DB ",
#                     )
#             try:
#                 # Generate responses for each question
#                 for question in questions:
#                     prompt = f"{question}. Provide an {'elaborate' if answer_type == 'elaborate' else 'specific'} answer."
#                     response = primary_model.generate_text(prompt, max_length=100)
#                     responses.append({"question": question, "answer": response})
#             except Exception as e:
#                 job_tracker.update_job_status(job_id, "Aborted")
#                 logger.error(
#                     f"Error in question_answering for one or more questions (job_id: {job_id}): {e}"
#                 )
#                 raise HTTPException(
#                     status_code=500,
#                     detail="Error in question_answering for one or more questions",
#                 )
#             logger.info(
#                 "Question and Answering job is now complete. Proceeding to cleanup activities for concluding the job."
#             )
#             # Move processed file
#             processed_file_path = os.path.join(PROCESSED_DIR, file.filename)
#             shutil.move(file_path, processed_file_path)

#             job_tracker.update_job_status(job_id, "Completed")
#             return {"job_id": job_id, "responses": responses}

#     except Exception as e:
#         job_tracker.update_job_status(job_id, "Aborted")
#         logger.error(f"Error in question_answering (job_id: {job_id}): {e}")
#         raise HTTPException(
#             status_code=500, detail="Failed to process question answering."
#         )

#     finally:
#         for file in files:
#             file_path = os.path.join(UPLOAD_DIR, file.filename)
#             if os.path.exists(file_path):
#                 os.remove(file_path)
#         torch.cuda.empty_cache()
#         gc.collect()
