import requests

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL_NAME = "qwen2.5:7b"

SYSTEM_PROMPT = """
너는 한국어로만 답변하는 요리 추천 챗봇이다.
반드시 한국어만 사용하라. 다른 언어는 절대 사용하지 마라.

아래에는:
- 사용자 요청
- 직전에 추천된 레시피 (있을 수도 있고 없을 수도 있음)
- 이번에 확정된 레시피 정보
가 주어진다.

중요 규칙:
- 이미 확정된 레시피만 설명하라.
- 다른 요리를 추천하지 마라.
- 비교하거나 대안을 제시하지 마라.
- 재료나 조리법을 추측하거나 추가하지 마라.

너의 역할:
- 이전 대화가 있다면 자연스럽게 이어서 설명하고
- 없다면 단독 추천처럼 설명하라.

출력은 한 문단의 자연스러운 한국어 문장만 허용된다.
"""

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

위 정보를 바탕으로
이번 레시피가 왜 사용자 요청에 잘 맞는지 설명해줘.
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

    return res.json()["choices"][0]["message"]["content"].strip()