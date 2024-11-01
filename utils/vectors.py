import os
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


class VectorManager:
    def __init__(self, config, device):
        self.config = config
        self.qdrant_client = QdrantClient(
            url=self.config["qdrant_local_url"], timeout=self.config["qdrant_timeout"]
        )
        """ self.qdrant_client.recreate_collection(
            collection_name=self.config["collection_name"],
            vectors_config=models.VectorParams(
                size=768, distance=models.Distance.COSINE
            ),
        ) """
        self.device = device
        os.makedirs(self.config["upload_dir"], exist_ok=True)

    def create_vector_store(
        self, documents: List[str], file_name: str, file_hash: str, config
    ) -> Qdrant:
        try:
            self.qdrant_client.get_collection(config["collection_name"])
        except Exception as e:
            if e.status_code == 404:
                # Create the collection if it doesn't exist
                self.qdrant_client.create_collection(
                    collection_name=config["collection_name"],
                    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
                )
            else:
                raise e
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="models/bge-m3",
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )
        vector_store = Qdrant(
            client=self.qdrant_client,
            embeddings=embeddings,
            collection_name=config["collection_name"],
        )

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
                    "file_hash": file_hash,
                }
                vector_store.add_documents([doc])

        return vector_store
