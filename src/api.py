# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from src.ranker import rank

app = FastAPI(
    title="AI Resume Ranker API",
    description="Ranks resumes based on Job Description using TF-IDF embeddings and skill matching",
    version="1.0"
)

class JDRequest(BaseModel):
    jd_text: str
    jd_skills: Optional[List[str]] = None
    top_k: int = 10

@app.get("/")
def root():
    return {"message": "Welcome to AI Resume Ranker API"}

@app.post("/rank")
def rank_resumes(payload: JDRequest):
    results = rank(payload.jd_text, jd_skills=payload.jd_skills, top_k=payload.top_k)
    return {"results": results}