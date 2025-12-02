def rule_adjust(tags, query):
    q = query.lower()

    
    if "찌개" in q:
        tags["category"] = ["찌개"]

    
    if any(k in q for k in ["칼칼", "얼큰", "얼큰한"]):
        tags["taste"] = ["매운"]
        if not tags["category"] or tags["category"] != ["찌개"]:
            tags["category"] = ["국-탕"]

    
    if tags["category"] == ["찌개"]:
        if "매운" in q:
            tags["taste"] = ["매운"]
        return tags

    if "매운" in q and any(k in q for k in ["국물", "탕", "찌개", "국"]):
        tags["category"] = ["국-탕"]
        tags["taste"] = ["매운"]

   
    if any(k in q for k in ["국물", "탕"]) or q.endswith("국") or " 국 " in q:
        if not tags["category"]:
            tags["category"] = ["국-탕"]

    if isinstance(tags["category"], list) and len(tags["category"]) > 1:
        priority = ["찌개", "국-탕", "면-만두", "밑반찬"]
        for p in priority:
            if p in tags["category"]:
                tags["category"] = [p]
                break

    if len(tags["taste"]) > 1:
        if "매운" in tags["taste"]:
            tags["taste"] = ["매운"]
        else:
            tags["taste"] = tags["taste"][:1]

    return tags