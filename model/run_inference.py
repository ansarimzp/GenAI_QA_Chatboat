from transformers import BertTokenizer, BertForQuestionAnswering
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import torch

MODEL_PATH = r"C:\Users\altam\Data_science\GenAI_QA_System\model\fine_tuned"
INDEX_PATH = r"C:\Users\altam\Data_science\GenAI_QA_System\model\faiss_index"

# Load model with error handling
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForQuestionAnswering.from_pretrained(MODEL_PATH)
except Exception as e:
    raise ValueError(f"Failed to load model from {MODEL_PATH}: {str(e)}")

# Load FAISS index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = FAISS.load_local(INDEX_PATH, embeddings)

def answer_question(question):
    try:
        # Retrieve context
        docs = db.similarity_search(question, k=3)
        context = " ".join([doc.page_content for doc in docs])
        
        # Tokenize inputs
        inputs = tokenizer(
            context,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Get predictions
        outputs = model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        
        # Decode answer
        answer = tokenizer.decode(
            inputs["input_ids"][0][answer_start:answer_end],
            skip_special_tokens=True
        )
        return answer.strip()
    
    except Exception as e:
        return f"Error processing question: {str(e)}"

        
        
