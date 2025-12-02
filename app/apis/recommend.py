from fastapi import APIRouter
from app.services.llm_client import analyze_text, normalize_tags
from app.services.rule_adjust import rule_adjust
from app.services.recommend_service import get_next_recipe
from app.services.session_manager import get_seen, add_seen, get_last_seen
from uuid import uuid4

router = APIRouter()

@router.get("/recommend")
def recommend(query: str, user_id: str):

    if user_id is None:
        user_id = f"guest-{uuid4()}"

    # 1) LLM 분석 (intent 포함)
    raw = analyze_text(query)
    intent = raw.get("intent", "query")  

    # 2) 태그 정리 + 규칙 적용
    tags = normalize_tags(raw)
    tags = rule_adjust(tags, query)

    # 3) 유저가 이미 본 레시피 목록
    seen_ids = get_seen(user_id)

    # 4) intent 처리 로직
    if intent == "reject":
        last_id = get_last_seen(user_id)
        if last_id:
            add_seen(user_id, last_id)   

  
    # 5) 새 추천 생성
    print(tags)
    recipe = get_next_recipe(tags, seen_ids)

    # 6) 추천된 레시피는 자동으로 seen 처리
    print("SEEN IDS BEFORE:", seen_ids)
    add_seen(user_id, recipe["recipe_id"])
    print("SEEN IDS AFTER:", get_seen(user_id))

 
    # 7) 응답 반환
    return {
        "intent": intent,
        "query": query,
        "user_id": user_id,
        "tags": tags,
        "seen": get_seen(user_id),
        "recipe": recipe
    }