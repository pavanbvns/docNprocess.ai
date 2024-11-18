<<<<<<< HEAD
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    MllamaForConditionalGeneration,
    AutoProcessor,
    pipeline,
    AutoModelForSequenceClassification,
)

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
=======
>>>>>>> ab8f96b (code without langchain, file summarization, qna on file)
import torch
import os
import gc
import logging
from transformers import (
    MllamaForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    # AutoModel,
    # AutoModelForImageClassification,
    BitsAndBytesConfig,
)
from config_loader import load_config  # Import directly from config_loader
from PIL import Image
import traceback


class ModelLoader:
    def __init__(self, config_file_path="config.yml"):
        self.config = load_config(config_file_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set CUDA memory management configuration for reduced fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        self.model, self.tokenizer, self.processor = self.model_loader()

    def model_loader(self):
        """Loads the Llama 3.2-11B Vision-Instruct Model with bitsandbytes quantization and CPU offloading."""
        try:
            logging.info(
                f"Loading {self.config["model_details"]["model_name"]} Model with quantization and CPU offloading..."
            )
            model_path = self.config["model_details"]["model_path"]

            # Set up bitsandbytes configuration for 8-bit quantization with CPU offloading enabled
            if self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    quant_type="nf4",  # Choose "nf4" or "fp4" as appropriate
                    llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offloading for 32-bit layers
                )
                device_map = "cuda"
                torch_dtype = torch.bfloat16

            else:
                quantization_config = None
                device_map = "cpu"
                torch_dtype = torch.bfloat16

            # Load model with quantization config and custom device map for CPU offloading
            model = MllamaForConditionalGeneration.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",  # Auto-dispatch layers to GPU or CPU as needed
                torch_dtype=torch.bfloat16,
            ).eval()

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            processor = AutoProcessor.from_pretrained(model_path)

            logging.info(
                f"{self.config["model_details"]["model_name"]} Model loaded successfully with CPU offloading."
            )
            return model, tokenizer, processor
        except Exception as e:
            logging.error(
                f"Failed to load {self.config["model_details"]["model_name"]} Model: {e}"
            )
            raise

    def generate_text(self, prompt_with_text, max_length=50):
        """Generates text from a prompt using the Llama model."""
        chunk_size = 2048
        chunks = [
            prompt_with_text[i : i + chunk_size]
            for i in range(0, len(prompt_with_text), chunk_size)
        ]
        intermediate_summaries = []
        try:
            for chunk in chunks:
                inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True).to(
                    self.device
                )
                with torch.no_grad():
                    output = self.model.generate(
                        inputs["input_ids"],
                        max_new_tokens=500,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.9,
                    )
                generated_text = self.tokenizer.decode(
                    output[0], skip_special_tokens=True
                )
                intermediate_summaries.append(generated_text)
                # Clear cache and collect garbage to optimize memory usage
            del inputs, output
            torch.cuda.empty_cache()
            gc.collect()

            # Combine intermediate summaries and generate final summary
            combined_text = " ".join(intermediate_summaries)
            inputs = self.tokenizer(combined_text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                final_summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=300,  # Target length for the final summary
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                )
            final_summary = self.tokenizer.decode(
                final_summary_ids[0], skip_special_tokens=True
            )
            return final_summary

        except Exception as e:
            logging.error(f"Failed to generate text: {e}")
            raise

    def process_image(self, image):
        """Processes an image for vision-based tasks using the Llama model."""
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model.generate(**inputs)

            return output
        except Exception as e:
            logging.error(f"Failed to process image: {e}")
            raise
        finally:
            # Clear inputs to free up memory after processing is complete
            del inputs
            torch.cuda.empty_cache()
            gc.collect()

    def get_text_embeddings(self, text):
        """Generates embeddings for a given text using bge-m3 if available, or fallback to Llama."""
        try:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embeddings = self.model(**inputs).logits.cpu().numpy()

            # Clear cache and collect garbage to optimize memory usage
            del inputs
            torch.cuda.empty_cache()
            gc.collect()

            return embeddings
        except Exception as e:
            logging.error(f"Failed to generate text embeddings: {e}")
            raise

    def get_image_embeddings(self, image):
        """Generates embeddings for an image using CLIP if available, or fallback to Llama."""
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embeddings = self.model(**inputs).logits

            # Clear cache and collect garbage to optimize memory usage
            del inputs
            torch.cuda.empty_cache()
            gc.collect()

            return embeddings.cpu().numpy()
        except Exception as e:
            logging.error(f"Failed to generate image embeddings: {e}")
            raise

    def generate_image_summary(
        self, image, summarization_depth="simple", min_words=50, max_words=500
    ):
        """
        Generates a summary for an image using the Vision model.

        Args:
            image (PIL.Image): The image to summarize.
            summarization_depth (str): Either "simple" or "thorough".
            min_words (int): Minimum word count for the summary.
            max_words (int): Maximum word count for the summary.

        Returns:
            str: The generated summary.
        """
        try:
            # Ensure summarization_depth is valid
            if summarization_depth not in ["simple", "thorough"]:
                raise ValueError("Summarization depth must be 'simple' or 'thorough'.")

            # Prepare the input for the model
            processor = self.processor
            inputs = processor(images=image, return_tensors="pt")

            # Generate the prompt based on summarization depth
            if summarization_depth == "simple":
                prompt = (
                    f"Summarize this image content in {min_words}-{max_words} words."
                )
            else:  # thorough
                prompt = (
                    f"Provide a detailed analysis of this image in {min_words}-{max_words} words, "
                    "including notable features and any potential text or visual data."
                )

            # Add prompt to the inputs
            inputs["input_ids"] = self.tokenizer(prompt, return_tensors="pt").input_ids

            # Generate summary
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_words,
                min_new_tokens=min_words,
                temperature=0.7,  # Adjust this for creative or factual summaries
                top_k=50,
            )
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary.strip()

        except Exception as e:
            logging.error(f"Error in generate_image_summary: {e}")
            raise RuntimeError(f"Failed to generate image summary: {e}")

    def answer_question_with_image(self, image, question, answer_type):
        """
        Answers a question based on the provided image using the Vision model.

        Args:
            image (PIL.Image.Image): The image to analyze.
            question (str): The question to answer based on the image.

        Returns:
            str: The answer generated by the Vision model.
        """
        try:
            if not isinstance(image, Image.Image):
                raise ValueError("Input is not a valid PIL Image.")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Question must be a non-empty string.")

            if answer_type == "specific":
                prompt = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {
                                "type": "text",
                                "text": f"""
                                        You are an assistant that provides precise and concise answers. 
                                        Answer only with the required value or data, nothing more.

                                        Question: {question}
                                        Answer:
                                        """,
                            },
                        ],
                    }
                ]
            elif answer_type == "elaborate":
                prompt = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {
                                "type": "text",
                                "text": f"""
                You are an assistant that provides detailed and descriptive answers.
                Explain the answer clearly and concisely.

                Question: {question}
                Answer:
                """,
                            },
                        ],
                    }
                ]
            else:
                raise ValueError(f"Invalid answer_type: {answer_type}")

            input_text = self.processor.apply_chat_template(
                prompt, add_generation_prompt=True
            )

            # Prepare inputs for the model
            inputs = self.processor(image, input_text, return_tensors="pt").to(
                self.device
            )
            if "input_ids" not in inputs or "pixel_values" not in inputs:
                raise ValueError("Processor failed to generate required inputs.")

            # Generate answer
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_return_sequences=1,
            )
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Find the index of "Answer:"
            answer_start_index = answer.find("Answer:") + len("Answer:")

            # Extract the text after "Answer:"
            answer = answer[answer_start_index:].strip()

            # Remove any leading/trailing "assistant" or other unwanted text
            answer = answer.replace("assistant", "").strip()

            print("answer: \n", answer)
            return answer
        except Exception as e:
            logging.error(
                f"Error answering question with image: {e} \n{traceback.format_exc()}"
            )
            raise RuntimeError("Image-based Q&A failed.")

    def answer_question_with_text(self, text, question, answer_type):
        """
        Answers a question based on the provided text using the Vision model.

        Args:
            text (str): The text to analyze.
            question (str): The question to answer based on the text.
            answer_type (str): The type of answer, either "specific" or "elaborate".

        Returns:
            str: The answer generated by the Vision model.
        """
        try:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Input text must be a non-empty string.")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Question must be a non-empty string.")

            # Prepare the prompt based on the answer type
            if answer_type == "specific":
                prompt = f"""
                You are an assistant that provides precise and concise answers.
                Answer only with the required value or data, nothing more.

                Text: {text}

                Question: {question}
                Answer:
                """
            elif answer_type == "elaborate":
                prompt = f"""
                You are an assistant that provides detailed and descriptive answers.
                Explain the answer clearly and concisely.

                Text: {text}

                Question: {question}
                Answer:
                """
            else:
                raise ValueError(f"Invalid answer_type: {answer_type}")

            # Tokenize the input prompt
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=4096
            ).to(self.device)

            if "input_ids" not in inputs:
                raise ValueError("Tokenizer failed to generate required inputs.")

            # Adjust max_length or use max_new_tokens
            max_tokens = min(len(inputs.input_ids[0]) + 50, 4096)
            # Generate the answer
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_tokens,
                num_return_sequences=1,
            )
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Find the index of "Answer:" to extract the relevant part
            answer_start_index = answer.find("Answer:") + len("Answer:")
            answer = answer[answer_start_index:].strip()

            # Remove any leading/trailing "assistant" or other unwanted text
            answer = answer.replace("assistant", "").strip()

            print("answer: \n", answer)
            return answer
        except Exception as e:
            logging.error(
                f"Error answering question with text: {e} \n{traceback.format_exc()}"
            )
            raise RuntimeError("Text-based Q&A failed.")
