import requests
import json

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL_NAME = "phi3:mini"
  

SYSTEM_PROMPT = """
너는 요리 추천 시스템을 위한 재료 추출기이다.
한국어로만 출력하고, JSON 외의 어떤 문장도 절대로 출력하지 마라.

출력해야 하는 JSON 필드는 다음 두 가지이다:

1) category — 아래 목록 중 하나 또는 빈 배열
2) ingredients — 사용자가 말한 문장에서 유추되는 요리 재료 키워드 목록
   재료는 실제 요리에 사용될 법한 단어만 포함하라.
   (예: 고추, 고춧가루, 청양고추, 대파, 마늘, 돼지고기, 육수, 두반장 등)

----------------------------------------
category 목록 (DB 기준)
["밑반찬", "메인반찬", "국-탕", "찌개", "면-만두",
 "밥-떡", "김치", "양식", "샐러드", "빵", "기타"]
----------------------------------------

재료 추출 규칙:
- 사용자가 원하는 음식의 맛/이미지/특징을 바탕으로
  실제로 자주 사용되는 재료를 추론하여 나열한다.
- 가능한 한 구체적인 단어를 사용한다. (예: "고기" 대신 "돼지고기")
- 판단이 어려우면 생략한다.
- JSON 외 다른 문장은 절대로 출력하지 마라.

출력 형식은 다음과 같다:

{
  "category": [],
  "ingredients": []
}
"""

USER_PROMPT_TEMPLATE = """
사용자 요청: "{user_query}"
위 요청을 위 JSON 형식으로만 출력해.
"""


def analyze_text(user_query: str) -> dict:
    user_prompt = USER_PROMPT_TEMPLATE.format(user_query=user_query)

    body = {
        "model": MODEL_NAME,
        "stream": False,
        "messages": [
            {"role" : "system", "content": SYSTEM_PROMPT},
            {"role" : "user", "content": user_prompt}
        ]
    }

    res = requests.post(OLLAMA_URL, json=body)
    res.raise_for_status()
    
    data = res.json()
    raw = data["choices"][0]["message"]["content"].strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        candidate = raw[start:end+1]
    else:
        candidate = raw

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {"error": "JSON parse failed", "raw": raw} 

def normalize_tags(raw: dict) -> dict:
    if not isinstance(raw, dict):
        return {
            "category": [],
            "taste": [],
            "temperature": "",
            "purpose": []
        }

    return {
        "category": raw.get("category", []) or [],
        "taste": raw.get("taste", []) or [],
        "temperature": raw.get("temperature", "") or "",
        "purpose": raw.get("purpose", []) or []
    }