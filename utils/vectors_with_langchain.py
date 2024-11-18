import hashlib
import threading
import logging
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, TextStreamer
from PIL import Image
from config_loader import load_config
from typing import List
import numpy as np
from langchain_qdrant import QdrantVectorStore
from langchain.llms.base import BaseLLM


class VectorStore:
    def __init__(
        self,
        model_loader,
        tokenizer=None,
        processor=None,
        config_file="config.yml",
    ):
        self.config = load_config(config_file)
        self.collection_name = self.config["qdrant"]["collection_name"]
        self.qdrant_client = self.initialize_qdrant_client()

        self.create_collection()  # Ensure collection setup
        self.model_loader = model_loader

        self.tokenizer = tokenizer

        # # Initialize the embedding model from the local path
        # self.embeddings_model = HuggingFaceBgeEmbeddings(
        #     model_name="models/bge-m3",
        #     model_kwargs={"device": "cpu", "trust_remote_code": True},
        #     encode_kwargs={"normalize_embeddings": True},
        # )

        # Initialize the vector store with the embeddings model
        self.qdrant_vector_store = self._initialize_vector_store()

    def initialize_qdrant_client(self):
        try:
            qdrant_url = self.config["qdrant"]["server_url"]
            client = QdrantClient(url=qdrant_url)
            logging.info(f"Connected to Qdrant server at {qdrant_url}")
            return client
        except Exception as e:
            logging.error(f"Failed to initialize Qdrant client: {e}")
            raise

    def create_collection(self):
        """Creates the Qdrant collection with specified parameters if it does not exist."""
        try:
            # Check if the collection already exists
            if not self.qdrant_client.collection_exists(self.collection_name):
                # Define vector parameters for the collection
                vector_size = self.config["qdrant"]["vector_size"]
                distance_metric = (
                    Distance.COSINE
                )  # Adjust if a different metric is desired

                # Create the collection with the specified vector size and distance metric
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size, distance=distance_metric
                    ),
                )
                logging.info(f"Collection '{self.collection_name}' created in Qdrant.")
            else:
                logging.info(
                    f"Collection '{self.collection_name}' already exists in Qdrant."
                )
        except Exception as e:
            logging.error(
                f"Failed to create or check collection '{self.collection_name}': {e}"
            )
            raise ValueError(
                f"Failed to create or check collection '{self.collection_name}': {e}"
            )

    def _initialize_vector_store(self):
        """Initializes the Qdrant vector store with an embedding function."""
        try:
            # Log embedding model path and collection name for tracking
            logging.info(
                "Initializing Qdrant vector store with local embeddings model."
            )

            # Initialize Qdrant vector store with embedding function
            self.qdrant_vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding=self.embeddings_model,
            )

            # Verify initialization
            if self.qdrant_vector_store is None:
                raise ValueError("Qdrant vector store failed to initialize.")

            logging.info(
                "Qdrant vector store initialized successfully with embeddings."
            )
            return self.qdrant_vector_store
        except Exception as e:
            logging.error(f"Failed to initialize vector store: {e}")
            raise ValueError(f"Failed to initialize vector store: {e}")

    def upsert_embedding(
        self, embedding, content_type, file_hash, file_uuid, content=None
    ):
        """Inserts a chunked embedding into Qdrant with content in payload."""
        try:
            if self.qdrant_vector_store is None:
                raise ValueError("Qdrant vector store not initialized.")
            payload = {
                "file_hash": file_hash,
                "file_uuid": file_uuid,
                "type": content_type,
                "content": content,  # Optional content summary
            }
            self.qdrant_vector_store.add_texts(
                texts=[content], metadatas=[payload], ids=[str(uuid4())]
            )
            logging.info(f"{content_type} embedding upserted successfully.")
        except Exception as e:
            logging.error(f"Failed to upsert {content_type} embedding: {e}")
            raise

    def query_embeddings_on_file(
        self, file_hash: str, query_list: List[str], top_k: int = 1
    ) -> List[str]:
        """
        Queries the Qdrant vector store directly using Qdrant.scroll and filters embeddings by file hash.
        Generates responses using the primary model loader.
        """
        try:
            answers = []
            for query in query_list:
                # Use text embedding model loader if available; fallback to primary model loader

                query_embedding = self.model_loader.get_text_embeddings(query)[0]

                # Filter to retrieve only points related to the specific file hash
                filter_condition = Filter(
                    must=[
                        FieldCondition(
                            key="file_hash", match=MatchValue(value=file_hash)
                        )
                    ]
                )

                # Retrieve embeddings using Qdrant.scroll
                results, next_offset = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    limit=top_k,
                    scroll_filter=filter_condition,
                    with_payload=True,
                )

                if not results:
                    answers.append(f"No relevant answers found for query: '{query}'")
                    continue

                # Rank results based on similarity to the query embedding
                ranked_results = sorted(
                    results,
                    key=lambda x: np.dot(query_embedding, x.vector)
                    / (np.linalg.norm(query_embedding) * np.linalg.norm(x.vector)),
                    reverse=True,
                )

                # Retrieve content of top-ranked result
                best_match = ranked_results[0].payload.get(
                    "content", "No content available"
                )

                # Generate a specific response using the model
                inputs = self.tokenizer(
                    f"Query: {query}\nContext: {best_match}",
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                )
                outputs = self.primary_model_loader.generate(
                    **inputs, max_length=256, num_beams=3, early_stopping=True
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                answers.append(response)

            logging.info(
                f"Successfully retrieved answers for {len(query_list)} questions."
            )
            print("the answer is: ", answers)
            return answers
        except Exception as e:
            logging.error(f"Failed to retrieve answers: {e}")

    def _check_existing_embedding(self, file_hash: str) -> bool:
        """Checks if an embedding with the specified file hash already exists in Qdrant."""
        try:
            # Set up a filter to find the document by file hash
            filter_condition = Filter(
                must=[
                    FieldCondition(key="file_hash", match=MatchValue(value=file_hash))
                ]
            )

            # Perform a search with the filter to find any matching points
            response = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1,
                scroll_filter=filter_condition,
                with_payload=False,
            )

            # If any point is found, return True; otherwise, False
            return len(response[0]) > 0 if response[0] else False
        except Exception as e:
            logging.error(f"Failed to check existing embedding: {e}")
            raise


# class CustomLLM(BaseLLM):
#     def __init__(self, model, tokenizer):
#         self.model = model
#         self.tokenizer = tokenizer

#     def _call(self, prompt: str, stop: List[str] = None) -> str:
#         inputs = self.tokenizer(
#             prompt, return_tensors="pt", max_length=2048, truncation=True
#         )
#         outputs = self.model.generate(**inputs)
#         return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
