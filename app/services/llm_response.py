import requests
import re

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL_NAME = "qwen2.5:7b"

SYSTEM_PROMPT = """
너는 한국어 사용자 전용 요리 추천 챗봇이다.

❗ 매우 중요:
- 출력에는 **한글과 한국어 문장부호만** 사용하라.
- 중국어, 영어, 한자, 일본어, 이모지, 외국어 단어를
  단 하나라도 포함하면 안 된다.
- 외국어가 떠오르더라도 반드시 **순수 한국어로 다시 바꿔서** 출력하라.
- 이 규칙을 어기면 잘못된 답변이다.

아래에는:
- 사용자 요청
- 직전 추천 레시피 (없을 수도 있음)
- 이번에 확정된 레시피 정보
가 주어진다.

중요 규칙:
- 이미 확정된 레시피만 언급하라.
- 다른 요리를 추천하거나 비교하지 마라.
- 재료나 조리법을 새로 추측하지 마라.
- 사용자의 요청을 다시 설명하지 마라.
- "원하셨는데", "요청에 부합합니다" 같은 메타 설명을 쓰지 마라.

출력 규칙:
- 추천 이유는 **자연스러운 한 문장**으로 작성한다.
- 설명문, 보고서체, 평가문처럼 쓰지 마라.
- 친구에게 말하듯 간단하게 말한다.
"""

def ensure_korean_only(text: str) -> str:
    """
    한글, 숫자, 공백, 기본 문장부호만 허용
    (외국어/한자/기타 기호 제거)
    """
    return re.sub(r"[^가-힣0-9\s.,!?~]", "", text)

def _summarize_recipe(recipe: dict) -> dict:
    """
    LLM #2에 넘길 레시피 요약 정보 생성
    """
    ingredients = recipe.get("ingredient", "")
    main_ings = [i.strip() for i in ingredients.split(",")[:4]]

    return {
        "name": recipe.get("name"),
        "main_ingredients": ", ".join(main_ings),
        "category": " / ".join(recipe.get("category", [])) if recipe.get("category") else ""
    }


def generate_response(
    user_query: str,
    recipe: dict,
    prev_recipe: dict | None = None
) -> str:

    if prev_recipe:
        context_text = f"""
[직전 추천 레시피]
이름: {prev_recipe["name"]}
"""
    else:
        context_text = "[직전 추천 레시피]\n없음"

    user_prompt = f"""
[사용자 요청]
{user_query}

{context_text}

[이번에 확정된 레시피]
이름: {recipe["name"]}
주요 재료: {recipe["ingredient"]}

위 정보를 참고해서
사용자에게 이 레시피를 가볍게 추천하는 한 문장을 써줘.
"""

    body = {
        "model": MODEL_NAME,
        "temperature": 0.2,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }

    res = requests.post(OLLAMA_URL, json=body)
    res.raise_for_status()

    content = res.json()["choices"][0]["message"]["content"].strip()
    content = ensure_korean_only(content)
    return content