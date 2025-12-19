import numpy as np
from app.services.embed_service import get_embedding
from models.recipe_loader_spring import get_all_recipes_from_spring

def build_recipe_vectors():
    recipes = get_all_recipes_from_spring()
    print(f"레시피 개수: {len(recipes)}")

    if not recipes:
        raise RuntimeError("❌ Spring에서 레시피를 불러오지 못했습니다.")

    vectors = []
    ids = []

    for r in recipes:
        text = " ".join([
            r.get("name", ""),
            r.get("ingredient", "") or "",
            r.get("spicyIngredient", "") or "",
            r.get("method", "") or ""
        ]).strip()

        if not text:
            continue

        emb = get_embedding(text)
        vectors.append(emb)
        ids.append(int(r["recipeId"]))  # 🔥 Spring 기준

    vectors = np.vstack(vectors)
    ids = np.array(ids, dtype=np.int32)

    np.save("models/recipe_vectors.npy", vectors)
    np.save("models/recipe_ids.npy", ids)

    print("✅ 임베딩 생성 완료:", vectors.shape)

if __name__ == "__main__":
    build_recipe_vectors()