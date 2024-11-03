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
from transformers import BitsAndBytesConfig
from PIL import Image

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
llm_loaded = Model.load_model_llama3211b(device, bnb_config)
tokenizer = Model.load_processor_llama3211b()
# classifier_model_loaded = Model.load_classifier_model()
# classifier_tokenizer = Model.load_classifier_tokenizer()
processor = Model.load_processor_llama3211b()


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
        llm_model=llm_loaded,
        tokenizer=tokenizer,
        embedding_model_name=config["embedding_model_path"],
        qdrant_url=config["qdrant_local_url"],
        collection_name=config["collection_name"],
    )
    Query = "Please generate summary of the document"
    doc_summary = chatbot.get_response(Query, file_uuid, file_hash)

    return doc_summary


@app.post("/qna_on_docs")
def qna_on_docs(uploaded_file: UploadFile = File(...), query: str = Query(...)):
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
        llm_model=llm_loaded,
        tokenizer=tokenizer,
        embedding_model_name=config["embedding_model_path"],
        qdrant_url=config["qdrant_local_url"],
        collection_name=config["collection_name"],
    )

    answer = chatbot.get_response(query, file_uuid, file_hash)

    return answer


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
