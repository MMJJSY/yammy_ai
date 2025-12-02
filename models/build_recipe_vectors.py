import numpy as np
from app.services.embed_service import get_embedding
from models.recipe_loader import get_all_recipes

def build_recipe_vectors():
    recipes = get_all_recipes()
    print(f"레시피 개수: {len(recipes)}")

    vectors = []
    ids = []

    for r in recipes:
        
        text_parts = [
            r.get("name", ""),
            r.get("ingredient", "") or "",
            r.get("spicy_ingredient", "") or "",
            str(r.get("method", "") or "")
        ]
        full_text = " ".join(text_parts)

        emb = get_embedding(full_text)
        vectors.append(emb)
        ids.append(int(r["recipe_id"]))

    vectors = np.vstack(vectors)  # (N, 768)
    ids = np.array(ids, dtype=np.int32)

    # recommend_service에서 이 경로를 읽고 있으니까 맞춰서 저장
    np.save("models/recipe_vectors.npy", vectors)
    np.save("models/recipe_ids.npy", ids)

    print("레시피 임베딩 생성 완료")
    print("shape:", vectors.shape)
    print("ids:", ids)


if __name__ == "__main__":
    build_recipe_vectors()