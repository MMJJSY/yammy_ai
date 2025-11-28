from fastapi import APIRouter
from app.services.llm_client import analyze_text

router = APIRouter()

@router.get("/analyze")
def analyze(query: str):
    return analyze_text(query)