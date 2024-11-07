from fastapi import FastAPI, APIRouter, UploadFile, File
import uvicorn
import yaml
from os import environ as env
import torch
import uuid
import hashlib
import shutil
from utils.vectors import VectorManager
from utils.chatbot import Chatbot_doc
from utils.model import Model
from utils.utils import utils
from transformers import BitsAndBytesConfig, pipeline
from PIL import Image
from pathlib import Path
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

app = FastAPI()

router = APIRouter()


with open("config.yml", "r") as conf:
    config = yaml.safe_load(conf)

# llm_loaded = Model.load_model_llama323b()
# tokenizer = Model.load_tokenizer_llama323b()
# classifier_model_loaded = Model.load_classifier_model()
# classifier_tokenizer = Model.load_classifier_tokenizer()

# Loading LLama 3.2 11B Vision Model, tokenizer, processor and pipeline
llama_llm_loaded = Model.load_model_llama3211b(device, bnb_config)
llama_tokenizer = Model.load_processor_llama3211b()
llama_processor = Model.load_processor_llama3211b()
# llama_text_gen_pipeline = pipeline(
#     "text-generation",
#     model=llama_llm_loaded,
#     tokenizer=llama_tokenizer,
#     max_new_tokens=3000,
# )


@router.get("/")
async def home():
    return {"message": f"Doc.ai service by {env['AUTHOR_NAME']}"}


VectorManager = VectorManager(config=config, device=device)


@app.post("/get_doc_summary")
def get_doc_summary(uploaded_file: UploadFile = File(...)):
    file_path = f"uploads/{uploaded_file.filename}"
    with open(file_path, "w+b") as file:
        shutil.copyfileobj(uploaded_file.file, file)

    file_uuid = str(uuid.uuid4())
    file_hash = hashlib.sha256(uploaded_file.file.read()).hexdigest()

    # load the saved file
    document_pages_list = utils.load_document(file_path)

    # create vectore store using the saved file
    qdrant_vectorstore = VectorManager.create_vector_store(
        documents=document_pages_list,
        file_name=uploaded_file.filename,
        file_hash=file_hash,
        config=config,
    )
    chatbot = Chatbot_doc(
        config=config,
        device=device,
        vector_store=qdrant_vectorstore,
        llm_model=llama_llm_loaded,
        tokenizer=llama_tokenizer,
        embedding_model_name=config["embedding_model_path"],
        qdrant_url=config["qdrant_local_url"],
        collection_name=config["collection_name"],
    )
    Query = "Please generate summary of the document"
    doc_summary = chatbot.get_response(Query, file_uuid, file_hash)

    return doc_summary


@app.post("/qna_on_docs")
def qna_on_docs(uploaded_file: UploadFile = File(...), query: str = Query(...)):
    file_name = uploaded_file.filename.strip(".")[0]
    Path(f"uploads/{file_name}/images_in_file").mkdir(parents=True, exist_ok=True)
    file_path = f"uploads/{file_name}/{uploaded_file.filename}"
    with open(file_path, "w+b") as file:
        shutil.copyfileobj(uploaded_file.file, file)

    file_uuid = str(uuid.uuid4())
    file_hash = hashlib.sha256(uploaded_file.file.read()).hexdigest()

    # load the saved file
    document_pages_list = utils.load_document(file_path)

    # create vectore store using the saved file
    qdrant_vectorstore = VectorManager.create_vector_store(
        documents=document_pages_list,
        file_name=uploaded_file.filename,
        file_hash=file_hash,
        config=config,
    )
    chatbot = Chatbot_doc(
        config=config,
        device=device,
        vector_store=qdrant_vectorstore,
        llm_model=llama_llm_loaded,
        tokenizer=llama_tokenizer,
        embedding_model_name=config["embedding_model_path"],
        qdrant_url=config["qdrant_local_url"],
        collection_name=config["collection_name"],
    )

    answer = chatbot.get_response(query, file_uuid, file_hash)

    return answer


<<<<<<< HEAD
=======
@app.post("/qna_doc_vision")
def qna_doc_vision(uploaded_file: UploadFile = File(...), query: str = Query(...)):
    foldername = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    Path(f"uploads/{foldername}/images_in_file").mkdir(parents=True, exist_ok=True)

    file_path = f"uploads/{foldername}/{uploaded_file.filename}"

    images_path = f"uploads/{foldername}/images_in_file"

    # file_path = f"uploads/{uploaded_file.filename}"
    with open(file_path, "w+b") as file:
        shutil.copyfileobj(uploaded_file.file, file)

    file_uuid = str(uuid.uuid4())
    file_hash = hashlib.sha256(uploaded_file.file.read()).hexdigest()

    if file_path.endswith(".pdf"):
        # process the document
        texts_in_doc, tables_in_doc = utils.process_document(file_path, images_path)

        # Prompt for generating descriptions
        prompt_text = """You are an assistant tasked with describing every row of the table. \
        Give a detailed description of every row of table. Table chunk: {element}. \
        Also give a summary of the table in the end."""
        # Generating descriptions for each table
        table_desc = []
        for text in tables_in_doc:
            inputs = llama_processor(
                None, prompt_text.format(element=text), return_tensors="pt"
            ).to(device)
            generated_text_encoded = llama_llm_loaded.generate(
                inputs.input_ids, max_length=500
            )
            generated_text = llama_processor.decode(generated_text_encoded[0])
            table_desc.append(generated_text)
        print("Tables description \n", table_desc)
        exit()
    elif (
        file_path.endswith(".jpg")
        or file_path.endswith(".png")
        or file_path.endswith(".tiff")
    ):
        image = Image.open(file_path)

    messageDataStructure = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": query}],
        }
    ]
    textInput = llama_processor.apply_chat_template(
        messageDataStructure, add_generation_prompt=True
    )

    inputs = llama_processor(image, textInput, return_tensors="pt").to(device)

    output = llama_llm_loaded.generate(**inputs, max_new_tokens=2000)

    generatedOutput = llama_processor.decode(output[0])

    print(generatedOutput)

    return generatedOutput


>>>>>>> 5677807 (progress on handling complex pdf / doc files)
""" @app.post("/chat_with_doc")
async def chat_with_doc(uploaded_file: UploadFile = File(...), query: str = Query(...)):
    # ... (rest of the code)

    # Generate a unique conversation ID
    conversation_id = str(uuid.uuid4())

    response = generate_response(vector_store, query, conversation_id)
    return {"answer": response} """


""" @app.post("/file_content_classifier")
def file_content_classifier(uploaded_file: UploadFile = File(...)):
    file_path = f"uploads/{uploaded_file.filename}"
    with open(file_path, "w+b") as file:
        shutil.copyfileobj(uploaded_file.file, file)

    # file_uuid = str(uuid.uuid4())
    # file_hash = hashlib.sha256(uploaded_file.file.read()).hexdigest()
    document_pages_list = utils.load_document(file_path)

    document_text = ""
    for page in document_pages_list:
        document_text = document_text + page.page_content
    # Classify a document
    print(document_text)
    # Tokenize the input text, truncating it to the maximum sequence length
    inputs = classifier_tokenizer(document_text, truncation=True, padding="max_length")

    # Convert the inputs to a PyTorch tensor
    inputs = {k: torch.tensor(v) for k, v in inputs.items()}

    # Pass the input to the model
    with torch.no_grad():
        outputs = classifier_model_loaded(**inputs)
        print(outputs)
        return outputs
 """

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("doc-ai_app:app", reload=True, port=6060, host="0.0.0.0")
