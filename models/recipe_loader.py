recipes = [
    {
        "id": 1,
        "name": "육개장",
        "category": "국물",
        "taste": "칼칼",
        "ingredients": ["양지", "고사리", "대파", "고추기름"],
        "steps": "소고기를 삶은 뒤 찢고, 고사리를 볶아 고추기름과 함께 끓인다."
    },
    {
        "id": 2,
        "name": "김치찌개",
        "category": "국물",
        "taste": "얼큰",
        "ingredients": ["김치", "돼지고기", "두부"],
        "steps": "돼지고기를 볶고 김치를 넣어 끓인 뒤 두부를 넣는다."
    },
    {
        "id": 3,
        "name": "짬뽕",
        "category": "국물",
        "taste": "매움",
        "ingredients": ["오징어", "홍합", "야채"],
        "steps": "해산물과 야채를 볶고 육수를 부어 끓인다."
    }
]

def load_all_recipes():
    """레시피 전체 반환"""
    return recipes

def get_recipe_by_id(recipe_id:int):
    """id로 레시피 검색"""
    for r in recipes:
        if r["id"] == recipe_id:
            return r
        
        return None
