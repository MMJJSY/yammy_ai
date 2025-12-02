import numpy as np
from numpy.linalg import norm
import random
from app.services.embed_service import get_embedding
from models.recipe_loader import get_recipe_by_id


# 로드된 임베딩 벡터들
recipe_vectors = np.load("models/recipe_vectors.npy")
recipe_ids = np.load("models/recipe_ids.npy")

# -----------------------------
# 1) 코사인 유사도
# -----------------------------
def cosine(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


# -----------------------------
# 2) 추천 후보 생성
# -----------------------------
def get_candidates(tags):
    TOTAL = len(recipe_vectors)

    # 전체의 10%, 범위: 20~100
    k = max(20, TOTAL // 10)
    k = min(k, 100)

    categories = tags.get("category", [])
    ingredients = tags.get("ingredients", [])

    # ✔ 재료 키워드 중심 Query 생성
    query_text = ""

    # category는 의미 반영 → 약하게 가중치
    if categories:
        query_text += (" ".join(categories) + " ") * 2

    # ingredients는 핵심 → 강하게 가중치
    if ingredients:
        query_text += (" ".join(ingredients) + " ") * 8

    # 백업용 한국 요리 공통 키워드
    query_text += "한국요리 국물 고기 고추마늘대파 야채 볶음 조림 찌개 국탕 "

    # 임베딩 생성
    query_vec = get_embedding(query_text)

    # 유사도 계산
    scores = [cosine(query_vec, v) for v in recipe_vectors]

    # top_k 반환
    top_indices = np.argsort(scores)[::-1][:k]

    return [int(recipe_ids[i]) for i in top_indices]


# -----------------------------
# 3) 최종 추천
# -----------------------------
def get_next_recipe(tags, seen_ids):
    candidates = get_candidates(tags)

    # 본 적 없는 레시피만
    filtered = [rid for rid in candidates if rid not in seen_ids]

    # 후보가 없으면 전체 후보에서 랜덤
    if not filtered:
        rid = random.choice(candidates)
        return get_recipe_by_id(rid)

    # 다양화를 위해 랜덤
    rid = random.choice(filtered)

    return get_recipe_by_id(rid)