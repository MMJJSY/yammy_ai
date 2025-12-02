import requests
import json

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL_NAME = "phi3:mini"
  

SYSTEM_PROMPT = """
너는 요리 추천 시스템을 위한 태그 분류기다.
항상 한국어로만 대답하고, JSON 외의 어떤 문장도 절대 출력하지 마라.

사용자 요청을 보고 아래 다섯 가지 태그를 선택하라.
모든 태그는 반드시 **아래 목록 중에서만** 고른다.
모르면 빈 배열([]) 또는 빈 문자열("")로 둔다.

========================================
📌 1) category (요리 종류 — type_category 테이블)
["밑반찬", "메인반찬", "국-탕", "찌개", "면-만두", 
 "밥-떡", "김치", "양식", "샐러드", "빵", "기타"]

📌 2) taste (맛)
["매운", "얼큰한", "짭짤한", "달콤한", "고소한", "새콤한", "담백한"]

📌 3) temperature (온도)
["hot", "cold", "warm"]

📌 4) purpose (목적)
["다이어트", "든든한", "해장", "야식", "간단"]
========================================

규칙:
- 위 목록에 없는 단어를 넣지 마라.
- 판단이 애매하면 비워둬라.
- JSON 외의 문장, 설명, 주석은 절대 넣지 마라.

출력 형식은 다음 JSON 한 개만:

{
  "category": [],
  "taste": [],
  "temperature": "",
  "purpose": []
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