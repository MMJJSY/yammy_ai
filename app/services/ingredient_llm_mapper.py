import requests
import json
from app.utils.json_guard import safe_json_array_parse


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = """
너는 요리 레시피 데이터베이스용 재료 정규화 도우미야.

규칙:
1. 입력은 사용자가 가진 재료 목록이다.
2. 출력은 레시피 DB에 들어갈 법한 '표준 재료명' 리스트다.
3. 수량, 단위, 형용사는 제거한다.
4. 동의어는 하나의 대표 재료명으로 통일한다.
5. JSON 배열만 출력한다.
"""

def normalize_ingredients_with_llm(user_ingredients: list[str]) -> list[str]:
    user_prompt = f"""
사용자 재료 목록:
{user_ingredients}

정규화된 재료 목록만 JSON 배열로 출력해.
"""

    body = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1
    }

    try:
        res = requests.post(OLLAMA_URL, json=body, timeout=20)
        res.raise_for_status()
        raw = res.json()["message"]["content"]
    except Exception as e:
        print("⚠ LLM 호출 실패:", e)
        return user_ingredients

    parsed = safe_json_array_parse(raw)

    if not parsed:
        print("⚠ JSON 파싱 실패, 원본 사용:", raw)
        return user_ingredients

    return parsed