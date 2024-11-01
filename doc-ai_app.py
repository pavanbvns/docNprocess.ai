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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()

router = APIRouter()


with open("config.yml", "r") as conf:
    config = yaml.safe_load(conf)

llm_loaded = Model.load_model_llama323b()
tokenizer = Model.load_tokenizer_llama323b()


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


app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("doc-ai_app:app", reload=True, port=6060, host="0.0.0.0")
