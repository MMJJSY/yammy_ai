import requests
import json

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL_NAME = "qwen2.5:7b"  

SYSTEM_PROMPT = """
너는 한국어만 사용하는 어시스턴트야. 어떤 상황에서도 한국어로만 대답해.

너는 요리 추천 시스템을 위한 해석기야.
사용자 요청을 태그 기반 JSON으로 변환해.

반드시 다음 형식의 JSON 한 개만 출력해.
설명 문장, 주석, 다른 텍스트는 절대로 출력하지 마.

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