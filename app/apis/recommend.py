from fastapi import APIRouter
from uuid import uuid4

from app.services.llm_client import analyze_text, normalize_tags
from app.services.rule_adjust import rule_adjust
from app.services.recommend_service import get_next_recipe
from app.services.session_manager import get_seen, add_seen

router = APIRouter()

@router.get("/recommend")
def recommend(query: str, user_id: str | None = None):

    if user_id is None:
        user_id = f"guest-{uuid4()}"

    # 1) LLM 태그 추출
    raw_tags = analyze_text(query)
    tags = normalize_tags(raw_tags)

    # 2) 룰 기반 보정 (국-탕, 면-만두, 재료 보강 등)
    tags = rule_adjust(tags, query)

    # 3) 이 유저가 지금까지 본 레시피 목록
    seen_ids = get_seen(user_id)

    # 4) 추천 생성 (Softmax + seen 제외 로직 들어있는 상태)
    recipe = get_next_recipe(query, tags, seen_ids)

    # 5) 본 레시피 기록
    if recipe and "recipe_id" in recipe:
        add_seen(user_id, recipe["recipe_id"])

    return {
        "query": query,
        "user_id": user_id,
        "tags": tags,
        "seen": get_seen(user_id),
        "recipe": recipe,
    }