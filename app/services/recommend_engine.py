import numpy as np
from numpy.linalg import norm
import random

from app.services.embed_service import get_embedding
from models.recipe_loader import (
    get_recipe_by_id,
    load_all_recipe_categories,
)
from app.utils.normalize import normalize_query





# 레시피 임베딩 로드
recipe_vectors = np.load("models/recipe_vectors.npy")   # (N, 768)
recipe_ids = np.load("models/recipe_ids.npy")           # (N,)


# --------------------------------------------------------
# Semantic booster (카테고리 의미 강화)
# --------------------------------------------------------
CATEGORY_KEYWORDS = {
    "밑반찬": "간단한 반찬 간단요리 무침 볶음 조림 짭짤한 집반찬",
    "메인반찬": "메인요리 고기 해물 든든한 구이 튀김 볶음 메인 디너",
    "국-탕": "국물 따뜻한 시원한 탕 깊은육수 한식국물 얼큰 개운한",
    "찌개": "찌개 얼큰 자작 국물 진한 맛 칼칼한 구수한 깊은맛 한식찌개",
    "면": "면요리 라면 칼국수 국수 우동 쫄깃한 면식",
    "파스타": "파스타 오일파스타 토마토파스타 크림파스타 양식 면요리 이탈리안",
    "밥": "밥 한식 백반 든든한 집밥 따뜻한 공기밥 기본식사",
    "볶음밥": "볶음밥 고슬고슬 볶은밥 한그릇요리 간단한 메뉴 볶음 맛있는",
    "덮밥": "덮밥 한그릇요리 밥위에 올린 음식 소스 든든한 덮어먹는 메뉴",
    "양식": "양식 버터 치즈 오븐 스테이크 수프 샐러드 서양식 요리",
    "샐러드": "샐러드 상큼 야채 건강식 가벼운 식사 드레싱 채소 신선한",
    "빵": "빵 토스트 샌드위치 베이커리 브런치 간단식 밀가루 버터 오븐",
    "떡볶이": "떡볶이 매운떡 국물떡볶이 분식 매콤한 쌀떡 밀떡 인기 간식",
    "간식": "간식 달달한 주전부리 과자 군것질 간단한 스낵",
    "디저트": "디저트 달콤한 케이크 쿠키 아이스크림 후식 브런치",
    "기타": "기타 요리 독특한 음식 단일메뉴 특별한요리",
}


# --------------------------------------------------------
# 🔥 재료 하드 필터용 함수 (핵심)
# --------------------------------------------------------
def recipe_contains_ingredients(recipe_id: int, ingredients: list[str]) -> bool:
    recipe = get_recipe_by_id(recipe_id)
    if not recipe:
        return False

    text = (
        (recipe.get("ingredient") or "") + " " +
        (recipe.get("spicy_ingredient") or "")
    )

    return all(ing in text for ing in ingredients)

# --------------------------------------------------------
# STEP 1. 후보 필터링 + query 강화
# --------------------------------------------------------
def get_candidates(user_query: str, tags: dict):
    ALL_CATEGORY_MAP = load_all_recipe_categories()

    categories = tags.get("category", []) or []
    ingredients = tags.get("ingredients", []) or []

    # 기본값: 전체
    filtered_ids = recipe_ids
    filtered_vecs = recipe_vectors


    
    # ----------------------------------------------------
    # 1) category + ingredient 하드 필터
    # ----------------------------------------------------
    if categories:
        target_cat = categories[0]

        new_ids = []
        new_vecs = []

        for rid, vec in zip(recipe_ids, recipe_vectors):

            cat_list = ALL_CATEGORY_MAP.get(str(rid), [])

            if categories:
                if not any(
                    target_cat in c or c in target_cat
                    for c in cat_list
                ):
                    continue

            if ingredients:
                if not recipe_contains_ingredients(rid, ingredients):
                    continue
            
            new_ids.append(rid)
            new_vecs.append(vec)

        if new_ids:
            filtered_ids = np.array(new_ids)
            filtered_vecs = np.array(new_vecs)

 
    # ----------------------------------------------------
    # 2) query_text 생성 (semantic boosting)
    # ----------------------------------------------------
    query_parts = []

    # 사용자 원문
    query_parts.append(user_query)

    # 카테고리 의미 강화
    if categories:
        key = CATEGORY_KEYWORDS.get(categories[0])
        if key:
            query_parts.append(key)

    # 재료 의미 강화 (가중치)
    if ingredients:
        ing_text = " ".join(ingredients)
        query_parts.append(ing_text)
        query_parts.append((ing_text + " ") * 3)

    # fallback
    if not query_parts:
        query_parts.append("요리 음식 레시피 한식 집밥")

    query_text = " ".join(query_parts)

    query_vec = get_embedding(query_text)

    # ----------------------------------------------------
    # 3) 유사도 계산
    # ----------------------------------------------------
    scores = np.dot(filtered_vecs, query_vec) / (
        norm(query_vec) * norm(filtered_vecs, axis=1)
    )

    TOP_K = 10
    k = min(TOP_K, len(scores))

    top_idx = np.argsort(scores)[::-1][:k]
    top_ids = list(filtered_ids[top_idx])
    top_scores = list(scores[top_idx])
    
    print("🔥 filtered_ids count:", len(filtered_ids))

    return top_ids, top_scores
   


    
# --------------------------------------------------------
# Softmax
# --------------------------------------------------------
def softmax(x):
    x = np.array(x, dtype=float)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_next_recipe(user_query: str, tags: dict, seen_ids):

    if tags.get("mode") == "fridge":
        return get_next_recipe_by_fridge(tags, seen_ids)
    
    user_query = normalize_query(user_query)
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

    if not filtered_ids:
        filtered_ids = candidates
        filtered_scores = scores

    # 🔥 최종 선택 ID 결정
    if len(tags.get("ingredients", [])) >= 2:
        rid = filtered_ids[0]
    else:
        probs = softmax(filtered_scores)
        rid = np.random.choice(filtered_ids, p=probs)

    # ================================
    # 🔥 여기!!!! (핵심 수정 포인트)
    # ================================
    recipe = get_recipe_by_id(rid)
    print("🔥 RETURN RECIPE =", recipe)
    if not recipe:
        return None

    # recipe_id 키 보장
    if "recipe_id" not in recipe and "id" in recipe:
        recipe["recipe_id"] = recipe["id"]

    return recipe

def get_next_recipe_by_fridge(tags: dict, seen_ids):
    ingredients = tags.get("ingredients", [])
    if not ingredients:
        return None

    INGREDIENT_MAP = {
        "고기": ["고기", "돼지고기", "소고기", "쇠고기", "닭", "닭고기"],
        "달걀": ["달걀", "계란"],
        "계란": ["달걀", "계란"],
        "파": ["파", "대파", "쪽파"],
        "고추": ["고추", "청양고추", "홍고추"],
        "면": ["면", "국수", "라면", "파스타", "짜파게티"],
        "밥": ["밥", "쌀"],
        "해산물": ["새우", "오징어", "조개", "게"],
    }

    def ingredient_match(text: str, ing: str) -> bool:
        candidates = INGREDIENT_MAP.get(ing, [ing])
        return any(c in text for c in candidates)

    scored = []

    for rid in recipe_ids:
        if rid in seen_ids:
            continue

        recipe = get_recipe_by_id(rid)
        if not recipe:
            continue

        text = (
            (recipe.get("ingredient") or "") + " " +
            (recipe.get("spicy_ingredient") or "")
        )

        # 🔹 재료 매칭 개수 (핵심)
        match_count = sum(
            ingredient_match(text, ing)
            for ing in ingredients
        )

        # ❌ 진짜 하나도 안 맞으면만 탈락
        if match_count == 0:
            continue

        # 🔹 면 요리 제약 (면 관련 재료 없으면 제외)
        if ingredient_match(text, "면") and not any(
            ing in ["면", "라면", "파스타"] for ing in ingredients
        ):
            continue

        # 🔹 점수 = 매칭 개수 (단순하고 안정적)
        score = match_count

        scored.append((score, rid))
        print("🧊 FRIDGE FILTER RESULT COUNT =", len(scored))

    if not scored:
        return None

    # 🔥 점수 높은 것 우선, 동점은 랜덤
    scored.sort(reverse=True)
    best_score = scored[0][0]
    top = [rid for s, rid in scored if s == best_score]

    rid = random.choice(top)
    recipe = get_recipe_by_id(rid)

    if recipe and "recipe_id" not in recipe and "id" in recipe:
        recipe["recipe_id"] = recipe["id"]

    print("🔥 FRIDGE RETURN RECIPE =", recipe)
    return recipe