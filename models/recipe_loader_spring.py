import requests
from typing import List, Dict

BACKEND_URL = "http://localhost:8080"

def get_all_recipes_from_spring():
    resp = requests.get("http://localhost:8080/api/recipes/all", timeout=20)
    resp.raise_for_status()
    return resp.json()