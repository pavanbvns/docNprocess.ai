# chatbot.py

# import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms.huggingface_pipeline import (
    HuggingFacePipeline,
)
from transformers import (
    TextStreamer,
    pipeline,
)

# from utils.model import Model
from utils.utils import utils
# from deleted.vectors import EmbeddingsManager
# from langchain_core.runnables.base import coerce_to_runnable


class Chatbot_doc:
    def __init__(
        self,
        config,
        vector_store,
        embedding_model_name: str,
        llm_model,
        qdrant_url: str,
        collection_name: str,
        tokenizer,
        device: str = utils.get_device_map(),
        encode_kwargs: dict = {"normalize_embeddings": True},
        llm_temperature: float = 0.7,
        # docs: list,
    ):
        """
        Initializes the ChatbotManager with embedding models, LLM, and vector store.

        Args:
            model_name (str): The HuggingFace model name for embeddings.
            device (str): The device to run the model on ('cpu' or 'cuda').
            encode_kwargs (dict): Additional keyword arguments for encoding.
            llm_model (str): The local LLM model name.
            llm_temperature (float): Temperature setting for the LLM.
            qdrant_url (str): The URL for the Qdrant instance.
            collection_name (str): The name of the Qdrant collection.
        """
        self.config = config
        self.embedding_model_name = embedding_model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        # self.llm_model = coerce_to_runnable(llm_model)
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.tokenizer = tokenizer
        # self.docs = docs
        self.vector_store = vector_store

        self.streamer = TextStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        # Initialize Local LLM pipeline
        self.text_gen_pipeline = pipeline(
            "text-generation",
            model=self.llm_model,
            # model="./models/Llama-3.2-3B-Instruct/",
            tokenizer=self.tokenizer,
            # temperature=self.llm_temperature,
            temperature=0.1,
            max_new_tokens=500,
            top_p=0.95,
            repetition_penalty=1.15,
            streamer=self.streamer,
            device=device,
            # Add other parameters if needed
        )
        self.llm = HuggingFacePipeline(pipeline=self.text_gen_pipeline)
        # self.llm = coerce_to_runnable(Model.load_model_llama323b)
        # Define the prompt template
        self.prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer need not be detailed or well explained or elaborated. If a specific question is being asked, then provide a very specifc answer and not a commentary
Helpful answer:
"""

        # Initialize Embeddings

        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )
        # Initialize Qdrant client
        self.client = QdrantClient(url=self.qdrant_url, prefer_grpc=False)

        # Initialize the prompt
        self.prompt = PromptTemplate(
            template=self.prompt_template, input_variables=["context", "question"]
        )

        # Initialize the retriever
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 1})

        # Define chain type kwargs
        self.chain_type_kwargs = {"prompt": self.prompt}

        # Initialize the RetrievalQA chain with return_source_documents=False
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            # llm="llama3",
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=False,  # Set to False to return only 'result'
            chain_type_kwargs=self.chain_type_kwargs,
            verbose=False,
        )

    def get_response(self, query: str, file_uuid, file_hash) -> str:
        """
        Processes the user's query and returns the chatbot's response.

        Args:
            query (str): The user's input question.

        Returns:
            str: The chatbot's response.
        """

        try:
            # response = self.qa.run(query)
            response = self.qa(query)
            result = response["result"]
            return result
        except Exception as e:
            return "⚠️ An error occurred while processing your request: \n" + str(e)
