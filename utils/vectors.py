import hashlib
import threading
import logging
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)
from PIL import Image
from typing import List
import numpy as np
import torch
from config_loader import load_config


class VectorStore:
    def __init__(
        self, model_loader, tokenizer=None, processor=None, config_file="config.yml"
    ):
        """
        Initializes the VectorStore with Llama 3.2 11B Vision as the embedding model.
        """
        self.model_loader = model_loader
        self.model = model_loader.model
        self.tokenizer = tokenizer or model_loader.tokenizer
        self.processor = processor or model_loader.processor
        self.device = model_loader.device

        self.config = load_config(config_file)
        self.qdrant_client = self.initialize_qdrant_client()
        self.collection_name = self.config["qdrant"]["collection_name"]
        self._initialize_collection()

    def initialize_qdrant_client(self):
        """Initializes the Qdrant client connected to the server."""
        try:
            qdrant_url = self.config["qdrant"]["server_url"]
            client = QdrantClient(url=qdrant_url)
            logging.info(f"Connected to Qdrant server at {qdrant_url}")
            return client
        except Exception as e:
            logging.error(f"Failed to initialize Qdrant client: {e}")
            raise

    def _initialize_collection(self):
        """Ensures the specified collection is created in Qdrant with vector settings."""
        try:
            if not self.qdrant_client.get_collection(self.collection_name):
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.config["qdrant"]["vector_size"],
                        distance=Distance.COSINE,
                    ),
                )
            logging.info(f"Collection '{self.collection_name}' initialized in Qdrant.")
        except Exception as e:
            logging.error(f"Failed to ensure collection '{self.collection_name}': {e}")
            raise ValueError(
                f"Failed to ensure collection '{self.collection_name}': {e}"
            )

    def _generate_embeddings(self, content: str) -> np.ndarray:
        """Generates embeddings for the given content using Llama 3.2 11B Vision."""
        try:
            inputs = self.tokenizer(content, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            return embeddings
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            raise

    def _check_existing_embedding(self, file_hash: str) -> bool:
        """Checks if an embedding for the given file hash exists in Qdrant."""
        try:
            filter_condition = Filter(
                must=[
                    FieldCondition(key="file_hash", match=MatchValue(value=file_hash))
                ]
            )
            response, _ = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1,
                scroll_filter=filter_condition,
                with_payload=False,
            )
            return len(response) > 0
        except Exception as e:
            logging.error(f"Failed to check existing embedding: {e}")
            raise

    def create_embeddings(self, files, file_type):
        """Creates embeddings for files if not already stored."""
        threads = []
        for file_path in files:
            file_hash = self._generate_hash(file_path)
            if self._check_existing_embedding(file_hash):
                logging.info(f"Embedding for {file_path} already exists. Skipping.")
                continue
            thread = threading.Thread(
                target=self._process_file, args=(file_path, file_type, file_hash)
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    def _process_file(self, file_path, file_type, file_hash):
        """Processes individual files to create embeddings based on file type."""
        try:
            file_uuid = str(uuid4())
            if file_type in ["pdf", "doc", "docx"]:
                self._process_text_file(file_path, file_hash, file_uuid)
            elif file_type in ["jpg", "jpeg", "png", "tiff"]:
                self._process_image_file(file_path, file_hash, file_uuid)
            else:
                logging.error(f"Unsupported file type: {file_type}")
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            raise

<<<<<<< HEAD
        # Split long documents and add to the vector store
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        for doc in documents:
            if len(doc.page_content) > 10000:  # Adjust the threshold as needed
                splits = text_splitter.split_text(doc.page_content)
                for split in splits:
                    split.metadata["custom_metadata"] = {
                        "file_name": file_name,
                        "file_hash": file_hash,
                    }
                vector_store.add_documents(splits)

            else:
                doc.metadata["custom_metadata"] = {
                    "file_name": file_name,
=======
    def _process_text_file(self, file_path, file_hash, file_uuid):
        """Creates text embeddings for document content and stores them in Qdrant."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            embeddings = self._generate_embeddings(content)
            point = PointStruct(
                id=str(uuid4()),
                vector=embeddings.tolist(),
                payload={
>>>>>>> ab8f96b (code without langchain, file summarization, qna on file)
                    "file_hash": file_hash,
                    "file_uuid": file_uuid,
                    "type": "text",
                    "content": content,
                },
            )
            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=[point], wait=True
            )
        except Exception as e:
            logging.error(f"Failed to process text file {file_path}: {e}")
            raise

    def _process_image_file(self, file_path, file_hash, file_uuid):
        """Creates image embeddings for an image file and stores them in Qdrant."""
        try:
            image = Image.open(file_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embeddings = (
                    self.model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
                )

            point = PointStruct(
                id=str(uuid4()),
                vector=embeddings.tolist(),
                payload={
                    "file_hash": file_hash,
                    "file_uuid": file_uuid,
                    "type": "image",
                },
            )
            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=[point], wait=True
            )
        except Exception as e:
            logging.error(f"Failed to process image file {file_path}: {e}")
            raise

    def query_embeddings(self, query_text=None, query_image=None, top_k=5):
        """Retrieves top-k nearest embeddings from Qdrant based on the input query."""
        try:
            if query_text:
                query_vector = self._generate_embeddings(query_text)[0]
            elif query_image:
                image = Image.open(query_image).convert("RGB")
                inputs = self.processor(images=image, return_tensors="pt").to(
                    self.device
                )
                with torch.no_grad():
                    query_vector = (
                        self.model(**inputs)
                        .last_hidden_state.mean(dim=1)
                        .cpu()
                        .numpy()[0]
                    )
            else:
                raise ValueError("Either query_text or query_image must be provided.")

            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=top_k,
            )
            return [
                {"id": result.id, "score": result.score, "payload": result.payload}
                for result in search_results
            ]
        except Exception as e:
            logging.error(f"Failed to query embeddings: {e}")
            raise

    def _generate_hash(self, file_path: str) -> str:
        """Generates a unique hash for the given file."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def upsert_embedding(self, elements_with_embeddings, file_path, job_id):
        """
        Inserts embeddings into Qdrant, preserving the sequence for accurate context retention.

        Args:
            elements_with_embeddings (list): List of elements with their embeddings and metadata.
            file_path (str): Path of the file for which embeddings are being created.
            job_id (str): Unique identifier for the processing job, used for exception handling and logging.
        """
        try:
            file_hash = self._generate_hash(
                file_path
            )  # Generate file hash for uniqueness
            file_uuid = str(uuid4())  # Create a unique identifier for the file

            points = []
            for element in elements_with_embeddings:
                payload = {
                    "file_hash": file_hash,
                    "file_uuid": file_uuid,
                    "type": element["type"],
                    "sequence_number": element["sequence_number"],
                    "content": element["content"],
                }

                points.append(
                    PointStruct(
                        id=str(uuid4()),
                        vector=element["embedding"],
                        payload=payload,
                    )
                )

            # Batch insert to Qdrant
            self._batch_insert_to_qdrant(points)
            logging.info(f"Embeddings for file '{file_path}' upserted successfully.")
        except Exception as e:
            logging.error(f"Failed to upsert embeddings for job '{job_id}': {e}")
            raise RuntimeError(
                f"Upsertion failed for job '{job_id}' due to an error: {e}"
            )

    def query_embeddings_on_file(self, file_path, questions, top_k=1):
        """
        Queries the Qdrant vector store for embeddings related to a specific file
        and retrieves answers for the given questions.

        Args:
            file_path (str): Path of the file for which embeddings are queried.
            questions (list): List of questions to query embeddings for.
            top_k (int): Number of top matches to retrieve for each question.

        Returns:
            list: List of responses containing question and corresponding answers.
        """
        try:
            file_hash = self._generate_hash(file_path)  # Generate file hash for lookup
            responses = []

            for question in questions:
                # Generate the query embedding for the question
                query_embedding = self.model_loader.get_text_embeddings(question)

                # Construct the filter to match the file hash
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="file_hash", match=MatchValue(value=file_hash)
                        )
                    ]
                )

                # Perform the search in the Qdrant vector store
                search_results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding[0],
                    limit=top_k,
                    with_payload=True,
                    filter=query_filter,
                )

                if not search_results:
                    responses.append(
                        {"question": question, "answer": "No relevant context found."}
                    )
                    continue

                # Extract content from top results
                context = " ".join(
                    [
                        result.payload.get("content", "")
                        for result in search_results.result
                    ]
                )

                # Generate the answer using the model
                prompt = f"Question: {question}\nContext: {context}\nAnswer:"
                inputs = self.model_loader.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(self.model_loader.device)

                outputs = self.model_loader.model.generate(
                    **inputs,
                    max_length=256,
                    num_return_sequences=1,
                )

                answer = self.model_loader.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                ).strip()

                responses.append({"question": question, "answer": answer})

            logging.info(
                f"Successfully retrieved answers for {len(questions)} questions."
            )
            return responses
        except Exception as e:
            logging.error(f"Failed to query embeddings for file '{file_path}': {e}")
            raise RuntimeError(f"Query failed due to an error: {e}")
