import time

# ==============================
# In-memory session storage
# ==============================
SESSION = {}
TTL = 60 * 60 * 12   # 12시간 (초)


def _cleanup(user_id: str):
    """
    TTL(12시간) 지난 seen 기록 제거
    """
    now = time.time()

    if user_id not in SESSION:
        return

    SESSION[user_id] = [
        item
        for item in SESSION[user_id]
        if now - item["seen_at"] < TTL
    ]

    if not SESSION[user_id]:
        del SESSION[user_id]


def get_seen(user_id: str) -> list[int]:
    """
    TTL 적용 후, recipe_id 리스트 반환
    """
    _cleanup(user_id)

    if user_id not in SESSION:
        return []

    return [item["recipe_id"] for item in SESSION[user_id]]


def add_seen(user_id: str, recipe_id: int):
    """
    새 recipe_id 기록
    """
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
        return seen[-1]   # dict 반환
    return None