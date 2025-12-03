import numpy as np
from numpy.linalg import norm
import random
from app.services.embed_service import get_embedding
from models.recipe_loader import get_recipe_by_id, load_all_recipe_categories

ALL_CATEGORY_MAP = load_all_recipe_categories()

# 레시피 임베딩 로드 (.npy 파일은 build_recipe_vectors.py에서 미리 생성)
recipe_vectors = np.load("models/recipe_vectors.npy")   # shape: (N, 768)
recipe_ids = np.load("models/recipe_ids.npy")           # shape: (N,)


def cosine(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


# category별 fallback 키워드
CATEGORY_KEYWORDS = {
    "국-탕": "국물 탕 찌개 얼큰 따뜻한 시원한 육수",
    "찌개": "찌개 얼큰 자작 양념 국물 육수 고춧가루",
    "면-만두": "면 면요리 칼국수 우동 라면 국물",
    "밑반찬": "반찬 볶음 조림 무침 자작 양념",
    "메인반찬": "고기 메인 요리 볶음 조림 구이",
    "양식": "치즈 버터 파스타 오븐 양식",
    "샐러드": "야채 상큼 드레싱 신선 가벼운",
    "빵": "빵 베이커리 밀가루 버터 오븐",
    "김치": "김치 배추 무 발효 고춧가루 마늘",
    "밥-떡": "밥 떡 든든한 한식 가정식",
    "기타": "요리 음식 한식 가정식",
}


def get_candidates(tags: dict):
    categories = tags.get("category", [])
    ingredients = tags.get("ingredients", [])

    # 기본값 = 모든 레시피
    filtered_ids = recipe_ids
    filtered_vecs = recipe_vectors

    # -----------------------------
    # 🔥 1) category 하드 필터링
    # -----------------------------
    if categories:
        target_cat = categories[0]

        new_ids = []
        new_vecs = []

        for rid, vec in zip(recipe_ids, recipe_vectors):
            if target_cat in ALL_CATEGORY_MAP.get(rid, []):
                new_ids.append(rid)
                new_vecs.append(vec)

        # 한 개라도 있으면 필터링된 집합만 사용
        if new_ids:
            filtered_ids = np.array(new_ids)
            filtered_vecs = np.array(new_vecs)

    # -----------------------------
    # 🔥 2) query_text 생성
    # -----------------------------
    query_parts = []

    # category fallback 단어
    if categories:
        key = CATEGORY_KEYWORDS.get(categories[0], "")
        query_parts.append(key)

    # ingredient 기반 강화
    if ingredients:
        ing = " ".join(ingredients)
        query_parts.append((ing + " ") * 10)

    if not query_parts:
        query_parts.append("요리 음식 국물 반찬 집밥 한식")

    query_text = " ".join(query_parts)
    query_vec = get_embedding(query_text)

    # -----------------------------
    # 🔥 3) 필터링된 집합에서만 SBERT 유사도 계산
    # -----------------------------
    scores = np.dot(filtered_vecs, query_vec) / (norm(query_vec) * norm(filtered_vecs, axis=1))

    k = min(50, len(scores))
    top_idx = np.argsort(scores)[::-1][:k]

    return list(filtered_ids[top_idx])

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_next_recipe(tags, seen_ids):
    candidates = get_candidates(tags)

    # 유사도 점수 다시 계산
    query_text = " ".join(tags.get("ingredients", []))
    query_vec = get_embedding(query_text)
    scores = np.array([cosine(query_vec, v) for v in recipe_vectors])

    # 후보 인덱스만 남기기
    candidate_indices = [np.where(recipe_ids == rid)[0][0] for rid in candidates]
    candidate_scores = scores[candidate_indices]

    # **softmax 가중치 적용 (확률 분포 생성)**
    weights = softmax(candidate_scores)

    # 이미 본 레시피 제거
    final_candidates = []
    final_weights = []

    for rid, w in zip(candidates, weights):
        if rid not in seen_ids:
            final_candidates.append(rid)
            final_weights.append(w)

    # 소진됐으면 전체 후보에서 softmax 랜덤 선택
    if not final_candidates:
        return get_recipe_by_id(np.random.choice(candidates, p=weights))

    # Softmax 가중 랜덤 선택
    rid = np.random.choice(final_candidates, p=np.array(final_weights)/sum(final_weights))
    return get_recipe_by_id(rid)