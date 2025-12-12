SYNONYM_MAP = {
    # 계란 계열
    "계란": "달걀",
    "에그": "달걀",

    # 파 계열 (원하면 추가)
    "파": "대파",

    # 고추 계열 (선택)
    "고추": "청양고추",
}

def normalize_query(q: str) -> str:
    for src, dst in SYNONYM_MAP.items():
        q = q.replace(src, dst)
    return q