from typing import List, Dict
import requests

BACKEND_URL = "http://localhost:8080"


def get_recipe_by_id(recipe_id: int) -> Dict | None:
    """
    Spring 백엔드에서 레시피 단건 조회
    """
    recipe_id = int(recipe_id)

    try:
        resp = requests.get(
            f"{BACKEND_URL}/api/recipes/{recipe_id}",
            timeout=3
        )
    except requests.RequestException as e:
        print(f"[ERROR] Backend request failed: {e}")
        return None

    if resp.status_code != 200:
        return None

    return resp.json()

def get_categories_by_recipe_id(recipe_id: int) -> List[str]:
    resp = requests.get(
        f"{BACKEND_URL}/api/recipes/{recipe_id}/categories",
        timeout=3
    )

    if resp.status_code != 200:
        return []

    return resp.json()   # ["국-탕", "찌개"]

def load_all_recipe_categories() -> Dict[int, List[str]]:
    resp = requests.get(
        f"{BACKEND_URL}/api/recipes/categories",
        timeout=5
    )

    if resp.status_code != 200:
        return {}

    return resp.json()

def get_all_recipes() -> List[Dict]:
    resp = requests.get(
        f"{BACKEND_URL}/api/recipes",
        timeout=5
    )

    if resp.status_code != 200:
        return []

    return resp.json()

