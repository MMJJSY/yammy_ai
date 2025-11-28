from fastapi import APIRouter
from app.services.llm_client import analyze_text
from app.services.recommend_service import find_best_recipe

router = APIRouter()

@router.get("/recommend")
def recommend(query: str):
    tags = analyze_text(query)

    recipe = find_best_recipe(tags)

    return {
        "query" : query,
        "tags" : tags,
        "recipe" : recipe
    }