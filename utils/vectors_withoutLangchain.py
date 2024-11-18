# import os
import hashlib
import threading
import logging
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    Batch,
    Distance,
    VectorParams,
    Filter,
    Match,
    FieldCondition,
    MatchValue,
)
from PIL import Image
from config_loader import load_config
from typing import List
import numpy as np


class VectorStore:
    def __init__(
        self,
        primary_model_loader,
        text_embedding_model_loader=None,
        image_embedding_model_loader=None,
        config_file="config.yml",
    ):
        """
        Initializes the VectorStore with a pre-loaded primary model.
        Optionally accepts text and image embedding models if use_full_pipeline is enabled.
        """
        self.config = load_config(config_file)
        self.qdrant_client = self.initialize_qdrant_client()
        self.primary_model_loader = primary_model_loader  # Pre-loaded primary model
        self.text_embedding_model_loader = text_embedding_model_loader
        self.image_embedding_model_loader = image_embedding_model_loader
        self.collection_name = self.config["qdrant"]["collection_name"]
        self._initialize_collection()

    def initialize_qdrant_client(self):
        """Initializes the Qdrant client connected to the running Qdrant server."""
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

    def _check_existing_embedding(self, file_hash):
        """Checks if an embedding for this file already exists in Qdrant."""
        try:
            # Set up filter to match the file_hash in payload
            file_filter = Filter(
                must=[
                    FieldCondition(key="file_hash", match=MatchValue(value=file_hash))
                ]
            )

            # Perform the scroll query with scroll_filter applied
            response, next_offset = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1,
                scroll_filter=file_filter,
                with_payload=True,  # Attach payload to verify result
            )

            # Check if any points were returned, indicating an existing embedding
            return len(response) > 0
        except Exception as e:
            logging.error(f"Failed to check existing embedding: {e}")
            raise

    def create_embeddings(self, files, file_type):
        """Creates embeddings for each file, if not already stored."""
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

    def _process_text_file(self, file_path, file_hash, file_uuid):
        """Creates text embeddings for document content and stores in Qdrant."""
        try:
            with open(file_path, "r") as file:
                content = file.read()

            if self.text_embedding_model_loader:
                text_embeddings = self.text_embedding_model_loader.get_text_embeddings(
                    content
                )
            else:
                text_embeddings = self.primary_model_loader.get_text_embeddings(content)

            points = [
                PointStruct(
                    id=str(uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "file_hash": file_hash,
                        "file_uuid": file_uuid,
                        "type": "text",
                        "content": content,
                    },
                )
                for embedding in text_embeddings
            ]

            self._batch_insert_to_qdrant(points)
        except Exception as e:
            logging.error(f"Failed to process text file {file_path}: {e}")
            raise

    def _process_image_file(self, file_path, file_hash, file_uuid):
        """Creates image embeddings for an image file and stores in Qdrant."""
        try:
            image = Image.open(file_path).convert("RGB")

            if self.image_embedding_model_loader:
                image_embeddings = (
                    self.image_embedding_model_loader.get_image_embeddings(image)
                )
            else:
                image_embeddings = self.primary_model_loader.get_image_embeddings(image)

            points = [
                PointStruct(
                    id=str(uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "file_hash": file_hash,
                        "file_uuid": file_uuid,
                        "type": "image",
                    },
                )
                for embedding in image_embeddings
            ]

            self._batch_insert_to_qdrant(points)
        except Exception as e:
            logging.error(f"Failed to process image file {file_path}: {e}")
            raise

    def _batch_insert_to_qdrant(self, points, batch_size=64):
        """Efficiently batch inserts points into Qdrant to minimize memory usage."""
        try:
            for i in range(0, len(points), batch_size):
                batch = Batch(points=points[i : i + batch_size])
                self.qdrant_client.upsert(
                    collection_name=self.collection_name, points=batch
                )
            logging.info(f"Inserted {len(points)} embeddings into Qdrant.")
        except Exception as e:
            logging.error(f"Error during batch insertion to Qdrant: {e}")
            raise

    def query_embeddings(self, query_text=None, query_image=None, top_k=5):
        """Retrieves top-k nearest embeddings from Qdrant based on the input query."""
        try:
            if query_text:
                query_vector = (
                    self.text_embedding_model_loader.get_text_embeddings(query_text)[0]
                    if self.text_embedding_model_loader
                    else self.primary_model_loader.get_text_embeddings(query_text)[0]
                )
            elif query_image:
                image = Image.open(query_image).convert("RGB")
                query_vector = (
                    self.image_embedding_model_loader.get_image_embeddings(image)[0]
                    if self.image_embedding_model_loader
                    else self.primary_model_loader.get_image_embeddings(image)[0]
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
                for result in search_results.result
            ]
        except Exception as e:
            logging.error(f"Failed to query embeddings: {e}")
            raise

    def upsert_embedding(
        self, embedding, content_type, file_hash, file_uuid, content=None
    ):
        """Inserts a chunked embedding into Qdrant with content in payload."""
        try:
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            if embedding.ndim > 1:  # If embedding is multi-dimensional
                embedding = embedding.flatten()
            embedding_data = embedding.tolist()

            payload = {
                "file_hash": file_hash,
                "file_uuid": file_uuid,
                "type": content_type,
                "content": content,  # Optional content summary
            }

            point = PointStruct(id=str(uuid4()), vector=embedding_data, payload=payload)

            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=[point], wait=True
            )
            logging.info("Embedding successfully upserted to Qdrant.")
        except Exception as e:
            logging.error(f"Failed to upsert {content_type} embedding: {e}")
            raise

    def query_embeddings_on_file(
        self, file_hash: str, query_list: List[str], top_k: int = 1
    ) -> List[str]:
        """Queries the Qdrant vector store with questions and retrieves answers.

        Args:
            file_hash (str): The unique hash of the file to query.
            query_list (List[str]): List of questions to get answers for.
            top_k (int): Number of top matches to retrieve for each question.

        Returns:
            List[str]: List of answers for each question in `query_list`.
        """
        answers = []

        try:
            # Iterate through each question in the query list
            for question in query_list:
                # Generate the query embedding for the question
                query_embedding = self.primary_model.get_text_embeddings(question)

                # Construct the filter to match the file hash
                query_filter = Filter(must=[Match(key="file_hash", value=file_hash)])

                # Perform the search in the Qdrant vector store
                search_results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding[0],
                    limit=top_k,
                    filter=query_filter,
                )

                # Extract the top answer based on search score
                if search_results:
                    # Take the content of the highest-scoring point
                    best_answer = search_results[0].payload.get("content")
                    answers.append(best_answer)
                else:
                    answers.append("No answer found")

            logging.info(
                f"Successfully retrieved answers for {len(query_list)} questions."
            )
            return answers

        except Exception as e:
            logging.error(f"Failed to query embeddings for questions: {e}")
            raise
