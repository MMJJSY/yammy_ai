import requests
import json
import re

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL_NAME = "qwen2.5:7b"  # 현재 쓰는 모델


SYSTEM_PROMPT = """
너는 요리 추천 시스템을 위한 '카테고리 + 재료 추출기'이다.

반드시 한국어로만 출력하고,
JSON 외의 어떤 문장도 절대로 출력하지 마라.

----------------------------------------
출력해야 하는 JSON 필드는 다음 두 가지이다:

1) category — 아래 목록 중 하나 또는 빈 배열([])
2) ingredients — 사용자가 원하는 음식에서 유추되는 재료들 (리스트)

   - 재료는 실제 요리에 들어갈 법한 "한국어 재료 이름"만 포함한다.
   - 예시:
     ["고춧가루", "고추장", "청양고추", "대파", "마늘", "양파", "배추",
      "돼지고기", "소고기", "닭고기", "두부", "애호박", "감자", "버섯",
      "계란", "멸치", "참치", "참기름", "들기름", "간장", "식용유", "설탕"]

   - 아래와 같은 것은 절대 넣지 마라:
     * 영어 단어 (예: "GROUND_PORK", "pork", "sauce")
     * 기호나 토큰 (예: ":'',", "''육수", "버무리기장" 같이 의미가 불명확한 것)
     * 레시피 전체 묘사, 문장, 문구

   - 재료는 짧은 명사 형태로만 작성한다.
     (예: "고수", "대파", "양파", "마늘", "고춧가루", "육수" 등)

----------------------------------------
category 목록 (DB 기준)

["밑반찬", "메인반찬", "국-탕", "찌개", "면-만두",
 "밥-떡", "김치", "양식", "샐러드", "빵", "간식", "디저트"]

----------------------------------------
재료 추출 규칙:

- 사용자의 요청을 보고 사용될 가능성이 높은 재료명을 추론해 나열한다.
- 가능한 한 구체적인 재료명을 사용한다.
  (예: "고기" 보다는 "돼지고기" 또는 "소고기"와 같이 구체적으로)
- 판단이 어려운 재료는 생략해도 된다.
- JSON 외 다른 문장, 설명, 주석을 절대 출력하지 마라.

----------------------------------------
반드시 아래 형식의 JSON "한 개"만 출력하라:

{
  "category": [],
  "ingredients": []
}
"""

USER_PROMPT_TEMPLATE = """
사용자 요청: "{user_query}"
위 요청을 보고, category와 ingredients를 위에서 설명한 JSON 형식으로만 출력하라.
"""


def analyze_text(user_query: str) -> dict:
    """
    Ollama(Qwen2.5:7B)에 user_query를 보내서
    category(List[str]), ingredients(List[str])를 추출한다.
    """
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