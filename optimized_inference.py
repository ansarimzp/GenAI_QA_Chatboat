from transformers import BertForQuestionAnswering, BertTokenizer
import torch
from torch.quantization import quantize_dynamic

MODEL_PATH = "C:/Users/altam/Data_science/GenAI_QA_System/model/fine_tuned"

# Load original model
model = BertForQuestionAnswering.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

# Quantize model
quantized_model = quantize_dynamic(
    model, 
    {torch.nn.Linear}, 
    dtype=torch.qint8
)

# Save quantized model
quantized_model.save_pretrained(MODEL_PATH + "_quantized")
tokenizer.save_pretrained(MODEL_PATH + "_quantized")
