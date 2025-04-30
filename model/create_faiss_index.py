from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import json

DATA_PATH = "C:/Users/altam/Data_science/GenAI_QA_System/data/qa_dataset.json"
INDEX_PATH = "C:/Users/altam/Data_science/GenAI_QA_System/model/faiss_index"

# Load dataset
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract contexts for indexing
contexts = [item["context"] for item in data]

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"}
)

# Create and save FAISS index
db = FAISS.from_texts(contexts, embeddings)
db.save_local(INDEX_PATH)
