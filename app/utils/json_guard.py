import json
import re

def safe_json_array_parse(text: str) -> list[str] | None:
    """
    LLM 출력에서 JSON 배열만 안전하게 추출
    실패 시 None 반환
    """
    if not text:
        return None

    # 1️⃣ 대괄호로 둘러싸인 부분만 추출
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return None

    candidate = match.group(0)

    # 2️⃣ trailing comma 제거
    candidate = re.sub(r",\s*\]", "]", candidate)

    try:
        data = json.loads(candidate)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        return None

    return None