def rule_adjust(tags: dict, query: str) -> dict:
    """
    LLM이 추출한 tags를
    의도 기반으로 제한적으로 보정한다.
    """
    category = tags.get("category", []) or []
    ingredients = tags.get("ingredients", []) or []

    q = query

    # -----------------------------
    # 1) 국물/찌개 카테고리 보정
    # -----------------------------
    if "찌개" in q:
        category = ["찌개"]
    elif any(k in q for k in ["국물", "국", "탕"]):
        if category != ["찌개"]:
            category = ["국-탕"]

    # -----------------------------
    # 2) 면 요리 보정
    # -----------------------------
    if any(k in q for k in ["라면", "국수", "칼국수", "우동", "냉면"]):
        category = ["면"]

    # -----------------------------
    # 3) 매운/칼칼/얼큰 → 재료 추론 (핵심)
    # -----------------------------
    if any(k in q for k in ["매운", "매콤", "칼칼", "얼큰"]):
        # 대표적인 '매운맛' 재료만 제한적으로 추가
        if "고춧가루" not in ingredients:
            ingredients.append("고춧가루")
        if "청양고추" not in ingredients:
            ingredients.append("청양고추")

    # -----------------------------
    # 4) 김치 명시 시
    # -----------------------------
    if "김치" in q and "김치" not in ingredients:
        ingredients.append("김치")

    # -----------------------------
    # 5) category가 여러 개면 우선순위 1개만
    # -----------------------------
    if len(category) > 1:
        priority = ["찌개", "국-탕", "면", "볶음밥", "덮밥"]
        picked = None
        for p in priority:
            if p in category:
                picked = p
                break
        category = [picked] if picked else category[:1]

    tags["category"] = category
    tags["ingredients"] = ingredients
    return tags