import time

SESSION = {}
TTL = 60 * 60 * 12   # 12시간 (초 단위)

def get_seen(user_id: str):
    """12시간 지난 항목은 자동 삭제, 나머지 recipe_id만 반환"""
    now = time.time()
    
    if user_id not in SESSION:
        return []

    # TTL(12시간) 지나면 제거
    SESSION[user_id] = [
        item for item in SESSION[user_id]
        if now - item["seen_at"] < TTL
    ]

    # recipe_id만 리스트로 반환
    return [item["recipe_id"] for item in SESSION[user_id]]


def add_seen(user_id: str, recipe_id: int):
    now = time.time()
    if user_id not in SESSION:
        SESSION[user_id] = []

    SESSION[user_id].append({
        "recipe_id": recipe_id,
        "seen_at": now
    })

def get_last_seen(user_id: str):
    seen = SESSION.get(user_id, [])
    if seen:
        return seen[-1]
    return None