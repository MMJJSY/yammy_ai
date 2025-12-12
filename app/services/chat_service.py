from app.services.llm_client import analyze_text          # LLM #1
from app.services.recommend_engine import get_next_recipe # SBERT
from app.services.llm_response import generate_response   # LLM #2


def chat_recommend(user_query: str, user_id: str, seen_ids: list):
    """
    챗봇용 추천 서비스
    - LLM #1: 의도 분석
    - SBERT: 레시피 결정
    - LLM #2: 자연어 응답 생성
    """

    # 1. LLM #1 — 사용자 의도 분석
    tags = analyze_text(user_query)

    # 2. SBERT — 레시피 결정
    recipe = get_next_recipe(user_query, tags, seen_ids)

    if not recipe:
        return {
            "answer": "조건에 맞는 요리를 찾지 못했어.",
            "recipe": None,
            "tags": tags
        }

    # 3. LLM #2 — 사람 말로 설명
    answer = generate_response(user_query, recipe)

    return {
        "answer": answer,
        "recipe": recipe,
        "tags": tags
    }