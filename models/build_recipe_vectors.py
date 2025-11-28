import numpy as np
from app.services.embed_service import get_embedding
from models.recipe_loader import load_all_recipes

recipes = load_all_recipes()

vectors = []
ids = []

for r in recipes:
    text = f"{r['name']} {r['category']} {r['taste']} {' '.join(r['ingredients'])} {r['steps']}"
    vec = get_embedding(text)

    vectors.append(vec)
    ids.append(r['id'])

np.save("models/recipe_vectors.npy", np.array(vectors))
np.save("models/recipe_ids.npy", np.array(ids))

print("레시피 임베딩 생성 완료")