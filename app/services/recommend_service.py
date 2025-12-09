import numpy as np
from numpy.linalg import norm
import random
from app.services.embed_service import get_embedding
from models.recipe_loader import get_recipe_by_id, load_all_recipe_categories

ALL_CATEGORY_MAP = load_all_recipe_categories()

# 레시피 임베딩 로드
recipe_vectors = np.load("models/recipe_vectors.npy")      # shape: (N, 768)
recipe_ids = np.load("models/recipe_ids.npy")              # shape: (N,)


def cosine(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


CATEGORY_KEYWORDS = {
    "국-탕": "국물 탕 시원한 얼큰한 따뜻한 깊은육수",
    "찌개": "찌개 얼큰 자작 국물 고춧가루 진한 깊은맛",
    "면-만두": "면 라면 칼국수 우동 소면 국수 만두 뜨끈한",
    "밑반찬": "반찬 볶음 조림 무침 간단 짭짤한",
    "메인반찬": "고기 메인요리 볶음 조림 구이 든든한",
    "밥-떡": "밥 한식 든든한 집밥 떡",
    "양식": "치즈 버터 파스타 오븐 크림 양식",
    "샐러드": "야채 상큼 드레싱 건강식",
    "빵": "빵 베이커리 밀가루 버터 오븐 달달한",
    "간식-디저트": "달콤 간식 디저트 아이스크림 시원한"
}


# --------------------------------------------------------
# 🔥 STEP 1. 후보 필터링 (카테고리 기반) + query 표현 강화
# --------------------------------------------------------
def get_candidates(user_query: str, tags: dict):
    categories = tags.get("category", [])
    ingredients = tags.get("ingredients", [])

    # 기본값: 전체 레시피
    filtered_ids = recipe_ids.copy()
    filtered_vecs = recipe_vectors.copy()

    # --------------------------------------------------------
    # 1) category 하드 필터링 (매우 중요)
    # --------------------------------------------------------
    if categories:
        target_cat = categories[0]

        new_ids = []
        new_vecs = []

        for rid, vec in zip(recipe_ids, recipe_vectors):
            if target_cat in ALL_CATEGORY_MAP.get(rid, []):
                new_ids.append(rid)
                new_vecs.append(vec)

        if new_ids:
            filtered_ids = np.array(new_ids)
            filtered_vecs = np.array(new_vecs)

    # --------------------------------------------------------
    # 2) query_text 생성 방식 개선 (네 품질 향상 핵심)
    # --------------------------------------------------------
    query_parts = []

    # (1) 사용자 입력 그대로 반영
    query_parts.append(user_query)

    # (2) category의 semantic keyword 보강
    if categories:
        key = CATEGORY_KEYWORDS.get(categories[0], "")
        if key:
            query_parts.append(key)

    # (3) ingredient 가중치 (×3 정도가 가장 안정적)
    if ingredients:
        ing_text = " ".join(ingredients)
        query_parts.append(ing_text)
        query_parts.append((ing_text + " ") * 3)

    # (4) fallback
    if len(query_parts) == 0:
        query_parts.append("요리 음식 맛있는 레시피 집밥 한식")

    query_text = " ".join(query_parts)

    # SBERT embedding
    query_vec = get_embedding(query_text)

    # --------------------------------------------------------
    # 3) 유사도 계산
    # --------------------------------------------------------
    scores = np.dot(filtered_vecs, query_vec) / (
        norm(query_vec) * norm(filtered_vecs, axis=1)
    )

    
    TOP_K = 10   # 데이터가 200~300개일 때 최적

    k = min(TOP_K, len(scores))
    top_idx = np.argsort(scores)[::-1][:k]

    top_ids = list(filtered_ids[top_idx])
    top_scores = list(scores[top_idx])

    return top_ids, top_scores


# --------------------------------------------------------
# Utility: softmax
# --------------------------------------------------------
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def is_strong_request(tags):
    # 재료 2개 이상이면 강한 요리 의도라고 판단
    ingredients = tags.get("ingredients", [])
    return len(ingredients) >= 2


# --------------------------------------------------------
# 🔥 STEP 2. 추천 로직 최종 결정
# --------------------------------------------------------
def get_next_recipe(user_query: str, tags: dict, seen_ids):
    candidates, scores = get_candidates(user_query, tags)

    if not candidates:
        return None

    # 이미 본 레시피 제거
    filtered_ids = []
    filtered_scores = []

    for rid, sc in zip(candidates, scores):
        if rid not in seen_ids:
            filtered_ids.append(rid)
            filtered_scores.append(sc)

    # 만약 후보가 모두 제거되면 기존 리스트 그대로 사용
    if not filtered_ids:
        filtered_ids = candidates
        filtered_scores = scores

    # --------------------------------------------------------
    # 강한 요청이면 → 무조건 Top1 반환
    # --------------------------------------------------------
    if is_strong_request(tags):
        rid = filtered_ids[0]
        return get_recipe_by_id(rid)

    # --------------------------------------------------------
    # 다양성을 위한 확률 기반 선택
    # --------------------------------------------------------
    probs = softmax(filtered_scores)
    rid = np.random.choice(filtered_ids, p=probs)

    return get_recipe_by_id(rid)