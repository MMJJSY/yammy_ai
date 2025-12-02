import numpy as np
from numpy.linalg import norm
from app.services.embed_service import get_embedding
from models.recipe_loader import get_recipe_by_id
from app.services.llm_client import normalize_tags

recipe_vectors = np.load("models/recipe_vectors.npy")
recipe_ids = np.load("models/recipe_ids.npy")

def cosine(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def find_best_recipe(tags: dict):
    tags = normalize_tags(tags)

    query_text = " ".join([
        " ".join(tags["category"]),
        " ".join(tags["taste"]),
        tags["temperature"],
        " ".join(tags["purpose"]),
    ]).strip()

    if not query_text:
        # LLM이 이상하게 나왔을 때 대비: 그냥 "한국 요리" 같은 기본 쿼리 넣기
        query_text = "한국 요리"

    query_vec = get_embedding(query_text)

    scores = [cosine(query_vec, v) for v in recipe_vectors]
    idx = int(np.argmax(scores))

    best_id = int(recipe_ids[idx])
    return get_recipe_by_id(best_id)
