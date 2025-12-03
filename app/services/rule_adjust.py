ALLOWED_CATEGORIES = [
    "밑반찬", "메인반찬", "국-탕", "찌개", "면-만두",
    "밥-떡", "김치", "양식", "샐러드", "빵", "기타"
]


def rule_adjust(tags: dict, query: str) -> dict:
    """
    LLM이 준 tags({ category: [...], ingredients: [...] })와
    원본 query를 보고 카테고리/재료를 룰로 보정.
    """
    q = query  # 한글 그대로 사용
    category = tags.get("category", []) or []
    ingredients = tags.get("ingredients", []) or []

    # 0) category 유효값만 남기기
    category = [c for c in category if c in ALLOWED_CATEGORIES]

    # 1) 국물/탕/찌개 관련 단어 → 국-탕/찌개 고정
    if "찌개" in q:
        category = ["찌개"]
    elif any(k in q for k in ["국물", "국", "탕"]):
        # 이미 찌개면 그대로 두고, 아니면 국-탕
        if category != ["찌개"]:
            category = ["국-탕"]

    # 2) 면/라면/우동/국수 → 면-만두 쪽
    if any(k in q for k in ["라면", "면요리", "면 요리", "우동", "국수", "칼국수", "소면"]):
        if not category:
            category = ["면-만두"]

    # 3) 김치 명시 → 김치 카테고리 우선
    if "김치" in q:
        category = ["김치"]

    # 4) category가 2개 이상이면 우선순위로 1개만 남기기
    if len(category) > 1:
        priority = ["찌개", "국-탕", "면-만두", "메인반찬", "밑반찬"]
        picked = None
        for p in priority:
            if p in category:
                picked = p
                break
        category = [picked] if picked else category[:1]

    # 5) 칼칼/얼큰 → 매운 재료 보정
    if any(k in q for k in ["칼칼", "얼큰", "맵게", "매콤"]):
        if "고춧가루" not in ingredients:
            ingredients.insert(0, "고춧가루")
        if "청양고추" not in ingredients:
            ingredients.insert(0, "청양고추")

    # 6) 국물 언급 시 육수 보정
    if any(k in q for k in ["국물", "국", "탕", "찌개"]) and "육수" not in ingredients:
        ingredients.append("육수")

    tags["category"] = category
    tags["ingredients"] = ingredients
    return tags