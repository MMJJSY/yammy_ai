import requests
import json
import re
from app.utils.normalize import normalize_query

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL_NAME = "qwen2.5:7b"  # 현재 쓰는 모델


SYSTEM_PROMPT = """
너는 요리 추천 시스템의 태그 추출기다.
반드시 JSON만 출력하라. JSON 이외의 문장, 설명, 주석은 절대 출력하지 마라.

목표:
- 사용자 문장에서 category 0~1개, ingredients 0~5개를 추출한다.

----------------------------------------
[허용 category 목록] (DB 기준)
["밑반찬","메인반찬","국-탕","찌개","면","파스타","밥","볶음밥","덮밥",
 "양식","샐러드","빵","떡볶이","간식","디저트","기타"]

규칙:
- category는 반드시 위 목록 중에서만 고른다.
- 확신이 없으면 category는 빈 배열 []로 둔다.
- category는 최대 1개만 출력한다.

----------------------------------------
[ingredients 규칙 — 매우 중요]
- ingredients에는 실제 식재료/양념 이름만 포함한다.
- 다음과 같은 추상적/일반 단어는 절대 포함하지 마라:
  ["재료","음식","요리","메뉴","국","탕","국물","찌개","면","밥","파스타",
   "볶음밥","덮밥","샌드위치","빵","디저트","간식","기타",
   "이런거","저런거","그런거","말고","추천",
   "매운거","매콤한거","칼칼한거"]

- 사용자가 명확히 언급한 재료만 추출한다.
- 재료가 명확히 언급되지 않으면 ingredients는 빈 배열 []로 둔다.
- 재료를 추측하거나 보완하지 마라.
- ingredients는 최대 5개, 중복 없이 출력한다.

----------------------------------------
[카테고리 추론 힌트]
- “찌개” → category=["찌개"]
- “국”, “탕”, “국물”이 음식 의미일 때 → category=["국-탕"]
- 라면/국수/칼국수/우동/냉면 → category=["면"]
- 알리오올리오/파스타 → category=["파스타"]
- 볶음밥 → category=["볶음밥"]
- 덮밥 → category=["덮밥"]
- 샌드위치/토스트 → category=["빵"]
- 떡볶이 → category=["떡볶이"]

----------------------------------------
[출력 형식]
{
  "category": [],
  "ingredients": []
}
"""

USER_PROMPT_TEMPLATE = """
사용자 요청: "{user_query}"

위 요청에서 category 0~1개, ingredients 0~5개를 추출해서
반드시 JSON만 출력하라.
"""


def analyze_text(user_query: str) -> dict:
    """
    Ollama(Qwen2.5:7B)에 user_query를 보내서
    category(List[str]), ingredients(List[str])를 추출한다.
    """
    user_query = normalize_query(user_query)
    
    user_prompt = USER_PROMPT_TEMPLATE.format(user_query=user_query)

    body = {
        "model": MODEL_NAME,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }

    res = requests.post(OLLAMA_URL, json=body)
    res.raise_for_status()

    data = res.json()
    raw = data["choices"][0]["message"]["content"].strip()

    # 혹시 앞뒤에 말이 붙어도 {} 블록만 추출
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        candidate = raw[start : end + 1]
    else:
        candidate = raw

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return {"error": "JSON parse failed", "raw": raw}

    return parsed


def _clean_ingredients_list(ing_list):
    """
    LLM이 뱉은 ingredients 리스트를 정제:
    - 문자열만 사용
    - 한글/숫자/공백 외 제거
    - 공백 기준으로 쪼개서 토큰화
    - 너무 짧거나 의미 없는 토큰 제거
    - 중복 제거
    """
    if not isinstance(ing_list, list):
        return []

    cleaned_tokens = []

    for item in ing_list:
        if not isinstance(item, str):
            continue

        s = item.strip()
        if not s:
            continue

        # 한글/숫자/공백만 남기기 (영어, 기호 제거)
        s = re.sub(r"[^가-힣0-9\s]", "", s)
        if not s:
            continue

        # 공백 기준으로 쪼개기
        tokens = [t.strip() for t in s.split() if t.strip()]
        for tok in tokens:
            # 숫자만 있는 토큰은 버린다 (예: "3", "200")
            if tok.isdigit():
                continue
            # 너무 짧은 한글(1글자) 토큰은 대부분 의미 없음 → 예외적으로 "파", "쑥" 같은 건 나중에 추가할 수 있음
            if len(tok) == 1 and tok not in ["파"]:
                continue
            cleaned_tokens.append(tok)

    # 순서 유지하면서 중복 제거
    final_list = []
    for t in cleaned_tokens:
        if t not in final_list:
            final_list.append(t)

    return final_list


def normalize_tags(raw: dict) -> dict:
    """
    LLM 응답(raw)을 정규화해서 항상

    {
        "category": [ ... ],
        "ingredients": [ ... ]
    }

    형태로 돌려준다.
    """
    result = {
        "category": [],
        "ingredients": [],
    }

    if not isinstance(raw, dict):
        return result

    # --- category 정규화 ---
    cat = raw.get("category", [])
    if isinstance(cat, str) and cat.strip():
        result["category"] = [cat.strip()]
    elif isinstance(cat, list):
        cleaned = []
        for c in cat:
            if isinstance(c, str) and c.strip():
                cleaned.append(c.strip())
        result["category"] = cleaned

    # --- ingredients 정규화 + 정제 ---
    ing = raw.get("ingredients", [])
    if isinstance(ing, str) and ing.strip():
        ing_list = [ing]
    else:
        ing_list = ing

    result["ingredients"] = _clean_ingredients_list(ing_list)

    return result