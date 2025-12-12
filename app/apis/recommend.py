from fastapi import APIRouter
from uuid import uuid4

from app.services.llm_client import analyze_text, normalize_tags
from app.services.llm_response import generate_response
from app.services.rule_adjust import rule_adjust
from app.services.recommend_engine import get_next_recipe
from app.services.session_manager import get_seen, add_seen, get_last_seen
from models.recipe_loader import get_recipe_by_id, load_all_recipe_categories

router = APIRouter()
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
    last_categories = ALL_CATEGORY_MAP.get(last_recipe_id, [])

    if last_categories:
        tags["category"] = [last_categories[0]]

    return tags


@router.get("/recommend")
def recommend(query: str, user_id: str | None = None):

    if user_id is None:
        user_id = f"guest-{uuid4()}"

    # 1️⃣ LLM1 태그 추출
    raw_tags = analyze_text(query)
    tags = normalize_tags(raw_tags)

    # 2️⃣ 규칙 보정
    tags = rule_adjust(tags, query)

    # 3️⃣ 🔥 대화 맥락 기반 카테고리 상속
    tags = inherit_previous_category(tags, query, user_id)

    # 4️⃣ 이전 추천 정보
    seen_ids = get_seen(user_id)
    last_seen = get_last_seen(user_id)
    prev_recipe = (
        get_recipe_by_id(last_seen["recipe_id"])
        if last_seen else None
    )

    # 5️⃣ 추천 엔진
    recipe = get_next_recipe(query, tags, seen_ids)

    if recipe and "recipe_id" in recipe:
        add_seen(user_id, recipe["recipe_id"])

    # 6️⃣ LLM2 응답 생성
    answer = None
    if recipe:
        answer = generate_response(
            user_query=query,
            recipe=recipe,
            prev_recipe=prev_recipe
        )

    return {
        "query": query,
        "user_id": user_id,
        "answer": answer,
        "tags": tags,
        "seen": get_seen(user_id),
        "recipe": recipe,
    }