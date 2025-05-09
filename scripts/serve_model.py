from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="ChildBright AI Pipeline")

class Query(BaseModel):
    prompt: str

# load once
chat = pipeline(
    'text-generation',
    model='models/qwen3-0.6b-lora',
    tokenizer='models/qwen3-0.6b-lora',
    device=0,
    max_new_tokens=100,
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(q: Query):
    out = chat(q.prompt)
    return {"response": out[0]['generated_text']}

