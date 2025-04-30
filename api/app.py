from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model.run_inference import answer_question

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def get_answer(request: QuestionRequest):
    try:
        response = answer_question(request.question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing request: {str(e)}"
        )
