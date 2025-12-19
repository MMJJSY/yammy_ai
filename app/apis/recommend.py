from fastapi import APIRouter
from uuid import uuid4

from app.services.llm_client import analyze_text, normalize_tags
from app.services.llm_response import generate_response
from app.services.ingredient_llm_mapper import normalize_ingredients_with_llm
from app.services.rule_adjust import rule_adjust
from app.services.recommend_engine import get_next_recipe
from app.services.session_manager import get_seen, add_seen, get_last_seen
from models.recipe_loader import load_all_recipe_categories
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()



# 🔥 레시피 ID → 카테고리 매핑 (AI 내부 판단용)
ALL_CATEGORY_MAP = load_all_recipe_categories()

# 🔥 후속 발화 판단 키워드
FOLLOWUP_KEYWORDS = ["말고", "더", "좀", "조금", "다른"]


def is_followup_query(query: str) -> bool:
    return any(k in query for k in FOLLOWUP_KEYWORDS)


def inherit_previous_category(tags: dict, query: str, user_id: str):
    """
    후속 발화이고 category가 비어 있으면
    이전 추천 레시피의 카테고리를 상속
    """
    if not is_followup_query(query):
        return tags

    last_seen = get_last_seen(user_id)
    if not last_seen:
        return tags

    last_recipe_id = last_seen["recipe_id"]
    last_categories = ALL_CATEGORY_MAP.get(str(last_recipe_id), [])

    if last_categories:
        tags["category"] = [last_categories[0]]

    return tags


@router.get("/recommend/chat")
def recommend_chat(query: str, user_id: str | None = None):

    if user_id is None:
        user_id = f"guest-{uuid4()}"

    raw_tags = analyze_text(query)
    tags = normalize_tags(raw_tags)

    tags = rule_adjust(tags, query)
    tags = inherit_previous_category(tags, query, user_id)

    seen_ids = get_seen(user_id)

    recipe = get_next_recipe(query, tags, seen_ids)

    # ✅ 여기서 recipeId 기준으로 검사
    if not recipe or not recipe.get("recipeId"):
        return {
            "user_id": user_id,
            "query": query,
            "recipe_id": None,
            "answer": "조건에 맞는 레시피를 찾지 못했어요.",
            "tags": tags
        }

    # ✅ recipeId로 꺼냄
    recipe_id = recipe["recipeId"]
    add_seen(user_id, recipe_id)

    answer = generate_response(
        user_query=query,
        recipe=recipe,
        prev_recipe=None
    )

    print("✅ FINAL RETURN recipe_id =", recipe_id)

    return {
        "user_id": user_id,
        "query": query,
        "recipe_id": recipe_id,   # 응답은 snake_case 유지 (Spring 친화)
        "answer": answer,
        "tags": tags
    }

class FridgeRecommendRequest(BaseModel):
    ingredients: List[str]
    user_id: Optional[str] = None
    
@router.post("/recommend/fridge")
def recommend_fridge(req: FridgeRecommendRequest):

    print("🧊 RAW REQ =", req)
    print("🧊 REQ.INGREDIENTS =", req.ingredients)


    user_id = req.user_id or f"guest-{uuid4()}"

    normalized_ingredients = normalize_ingredients_with_llm(req.ingredients)
    print("🧊 NORMALIZED INGREDIENTS =", normalized_ingredients)
    tags = {
        "mode": "fridge",   # 🔥 반드시 필요
        "category": [],
        "ingredients": normalized_ingredients
    }

    seen_ids = get_seen(user_id)

    recipe = get_next_recipe(
        user_query="냉장고 재료 기반 추천",
        tags=tags,
        seen_ids=seen_ids
    )

    

    # 🔥🔥🔥 여기서 키 정규화 (핵심)
    if recipe and "recipe_id" not in recipe:
        if "recipeId" in recipe:
            recipe["recipe_id"] = recipe["recipeId"]
        elif "id" in recipe:
            recipe["recipe_id"] = recipe["id"]

    print("🔥 FINAL FRIDGE RECIPE =", recipe)

    if not recipe or not recipe.get("recipe_id"):
        return {
            "user_id": user_id,
            "recipe_id": None,
            "answer": "해당 재료로 만들 수 있는 레시피를 찾지 못했어요.",
            "tags": tags
        }

    recipe_id = recipe["recipe_id"]
    add_seen(user_id, recipe_id)

    answer = generate_response(
    user_query="냉장고 재료로 추천",
    recipe=recipe,
    prev_recipe=None,
    mode="fridge",
    fridge_ingredients=normalized_ingredients
)

    return {
        "user_id": user_id,
        "recipe_id": recipe_id,
        "answer": answer,
        "tags": tags
    }