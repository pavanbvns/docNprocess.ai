from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    MllamaForConditionalGeneration,
    AutoProcessor,
    pipeline,
    AutoModelForSequenceClassification,
)

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import torch
# from langchain_community.llms import CTransformers


class Model:
    """A model class to lead the model and tokenizer"""

    def __init__(self) -> None:
        pass

    def load_model_llama323b():
        model = AutoModelForCausalLM.from_pretrained("./models/Llama-3.2-3B")
        return model

    def load_tokenizer_llama323b():
        tokenizer = AutoTokenizer.from_pretrained("./models/Llama-3.2-3B")
        return tokenizer

    # bnb_config
    def load_model_llama3211b(device, bnb_config):
        model = MllamaForConditionalGeneration.from_pretrained(
            "./models/Llama-3.2-11B-Vision-Instruct",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        return model

    def load_processor_llama3211b():
        processor = AutoProcessor.from_pretrained(
            "./models/Llama-3.2-11B-Vision-Instruct"
        )
        return processor

    def create_embeddings(device):
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="./models/bge-m3/",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        return embeddings

    """ def load_llama323b():
        # Load the locally downloaded model here
        llm = CTransformers(
            model="./models/Llama-3.2-3B-Instruct/", model_type="llama", temperature=0.7
        )
        return llm """

    def load_classifier_model():
        classifier_model = AutoModelForSequenceClassification.from_pretrained(
            "./models/distilbert-base-cased-distilled-squad"
        )
        return classifier_model

    def load_classifier_tokenizer():
        classifier_tokenizer = AutoTokenizer.from_pretrained(
            "./models/distilbert-base-cased-distilled-squad"
        )

        return classifier_tokenizer
