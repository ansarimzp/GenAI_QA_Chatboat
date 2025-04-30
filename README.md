# 🤖 GenAI Q&A Chatbot System

This repository hosts an intelligent and scalable **Question-Answering (QA) chatbot system** that combines the strengths of **transformer-based models** and **retrieval-augmented generation (RAG)** to provide context-aware, accurate responses to natural language queries.

Designed for real-world applications in customer service, knowledge bases, education, and enterprise automation, this project seamlessly integrates deep learning with vector search to achieve high performance and reliability.

---

## 🔍 Why This Project?
🧩 The Problem: Pretrained AI Isn’t Always Enough
Pretrained language models like BERT and GPT are powerful, but they have a limitation:

They only "know" what they were trained on — and that training data has a cutoff.

In real-world use cases like customer support, academic help, or domain-specific knowledge (e.g., healthcare, legal, enterprise), answers must be accurate, up-to-date, and based on external documents.
Out-of-the-box models can hallucinate, give vague answers, or ignore context completely.

## 💡 The Solution: Build a Smart Q&A System with Context
To solve this, I built a Retrieval-Augmented Generation (RAG)-based Question Answering System from scratch. Here's how I approached the solution:

### ✅ Step 1: Fine-Tuned a BERT Model
I started with BERT (a pre-trained transformer) and fine-tuned it on a custom Q&A dataset. This helped the model:

Understand my domain better (not just general English)

Improve the precision of answer extraction from context

### ✅ Step 2: Created a Semantic Search Engine with FAISS
I used sentence-transformers to turn every document/context into a vector — a mathematical representation of its meaning.
Then I stored those in a FAISS index, so when a user asks a question, the system:

Finds the top matching contexts

Sends them to the QA model as supporting evidence

### ✅ Step 3: Integrated It All with FastAPI
To make it usable, I wrapped everything in a FastAPI backend. Now users can:

Ask questions via API or frontend

Get fast, accurate answers in real-time

### ✅ Step 4: Performance Optimization
To make it lightweight and production-ready:

I applied dynamic quantization to reduce the model size and speed up inference

Ensured it supports scalable batch processing

## 🔨 What I Did in This Project
This wasn’t just plug-and-play — I actually engineered each layer:


| Component                     | What I Did                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| **Fine-tuning**              | Curated dataset, encoded QA pairs, trained and saved BERT-based QA model   |
| **Semantic Search**          | Used LangChain + FAISS to create and load a fast vector database           |
| **Backend Inference Logic**  | Wrote custom logic to connect search → context → BERT answer               |
| **Optimization**             | Quantized model for lower latency and smaller memory footprint             |
| **API Integration**          | Built FastAPI app with endpoints for QA + optional Streamlit interface     |
| **Documentation & ReadMe**   | Structured the project, wrote clear instructions, and explained architecture |

## 🧠 In Simple Terms:
"You ask a question → the system finds relevant documents → the AI model reads them → it gives you a precise answer — just like a smart assistant who knows where to look."
---

## ✨ Key Features

✅ Fine-tuned BERT for domain-specific QA  
✅ Retrieval-Augmented Generation for grounded answers  
✅ FAISS-based fast semantic context retrieval  
✅ API-powered and optionally UI-enabled deployment  
✅ Quantization-enabled model for faster inference  
✅ Modular, well-documented codebase  

---

## 🧠 System Architecture

```
                   +------------------------+
                   |   User Asks Question   |
                   +-----------+------------+
                               |
                               v
           +------------------+-------------------+
           |  Semantic Search (FAISS + Embeddings) |
           +------------------+-------------------+
                               |
                      Retrieved Context
                               |
                               v
           +------------------+-------------------+
           |  BERT QA Model (Fine-tuned)          |
           +------------------+-------------------+
                               |
                          Generated Answer
                               |
                               v
                   +------------------------+
                   |   Delivered to User    |
                   +------------------------+
```

This pipeline ensures both speed and accuracy by blending **retrieval-based context** with **transformer-powered reasoning**.

---

## 🛠️ Technologies Used

| Tool/Library           | Role                                      |
|------------------------|-------------------------------------------|
| `transformers`         | Pre-trained BERT model + QA fine-tuning   |
| `sentence-transformers`| High-quality sentence embeddings          |
| `faiss`                | Approximate nearest neighbor search       |
| `langchain`            | Vector store management and retrieval     |
| `pytorch`              | Training and inference                    |
| `FastAPI`              | Real-time backend API                     |
| `Streamlit`            | Optional web interface                    |
| `torch.quantization`   | Inference acceleration                    |

---

## 📁 Project Structure

```
GenAI_QA_System/
├── data/
│   └── qa_dataset.json             # Custom Q&A training data
│
├── model/
│   ├── fine_tune.py               # Fine-tune BERT script
|   ├── run_inference.py           # Main RAG-based inference
│
├── api/
│   └── app.py                    # FastAPI entrypoint
└── optimized_inference.py         # Faster inference with quantized model
|── create_faiss_index.py          # Build FAISS vector store
├── requirements.txt               # All dependencies
└── README.md                      # You’re reading it!
```

---

## 🚀 Quick Start

### ✅ Installation

```bash
git clone https://github.com/ansarimzp/GenAI_QA_System.git
cd GenAI_QA_System
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 🎓 Model Training & Index Creation

```bash
# Fine-tune the model on your custom dataset
python src/fine_tune.py

# Build the FAISS index
python src/create_faiss_index.py
```

### 🚦 Run the Backend API

```bash
uvicorn app.main:app --reload
```

Access the API at `http://localhost:8000`

---

## 📡 API Usage

### `POST /ask`

```json
{
  "question": "What does BERT stand for?"
}
```

**Response:**
```json
{
  "answer": "BERT stands for Bidirectional Encoder Representations from Transformers.",
  "confidence": 0.94
}
```

---

## 💬 Interactive Streamlit UI

```bash
streamlit run app/ui.py
```

Navigate to `http://localhost:8501` in your browser and start asking questions with a simple interface.

---

## 🔧 Performance Optimization

- ✅ **Model Quantization** using `torch.quantization` to reduce memory and boost speed
- ⚡ **FAISS-based Retrieval** for sub-second document search
- ♻️ **Context Caching** and **batched tokenization** to reduce compute time

---

## 🧪 Customize with Your Data

You can fine-tune the system to your domain:

1. Prepare a JSON dataset:
```json
[
  {
    "context": "Tesla was founded in 2003 by engineers...",
    "question": "When was Tesla founded?",
    "answer": "2003"
  }
]
```

2. Fine-tune:
```bash
python src/fine_tune.py --data_path ./data/your_dataset.json
```

3. Re-index:
```bash
python src/create_faiss_index.py --data_path ./data/your_dataset.json
```

---

## 🌱 Future Enhancements

- ☑️ Support for multilingual QA
- 🔄 Conversational memory with history
- 📈 Real-time feedback loop for answer quality
- 🐳 Docker-based deployment for portability
- 🤝 Integration with vector-capable databases (e.g., Weaviate, Pinecone)

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- 🤗 Hugging Face for providing transformer models
- 🧠 LangChain for chaining logic and vector store wrappers
- 🔍 Facebook Research for FAISS
- 🚀 The open-source community for continued innovation

---

Ready to build your own domain-specific AI Q&A engine?  
**Fork this project, fine-tune with your knowledge base, and deploy it in minutes.**

---
```

---

Let me know if you'd like me to generate this as a downloadable `README.md` file or help create the `ui.py` Streamlit interface.
