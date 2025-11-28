import numpy as np
from numpy.linalg import norm
from app.services.embed_service import get_embedding
from models.recipe_loader import get_recipe_by_id

recipe_vectors = np.load("models/recipe_vectors.npy")
recipe_ids = np.load("models/recipe_ids.npy")

def cosine(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def find_best_recipe(tags:dict):
    query_text = (
        " ".join(tags["category"]) + " " +
        " ".join(tags["taste"]) + " " + 
        tags.get("temperature", "")
    )

    query_vec = get_embedding(query_text)

    scores = [cosine(query_vec, v) for v in recipe_vectors]
    idx = int(np.argmax(scores))

    best_id = int(recipe_ids[idx])
    return get_recipe_by_id(best_id)
