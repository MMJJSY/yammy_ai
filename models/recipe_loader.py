from typing import List, Dict
from app.db import get_connection

def _row_to_dict(columns, row):
    result = {}
    for col, value in zip(columns, row):
        col = col.lower()
        if hasattr(value, "read"):   # CLOB인 경우
            result[col] = value.read()
        else:
            result[col] = value
    return result

def get_recipe_by_id(recipe_id: int) -> Dict | None:
    recipe_id = int(recipe_id)
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            recipe_id,
            name,
            serving,
            time,
            ingredient,
            spicy_ingredient,
            method
        FROM recipe
        WHERE recipe_id = :id
    """, {"id": recipe_id})

    row = cur.fetchone()

    if not row:
        cur.close()
        conn.close()
        return None

    columns = ["recipe_id", "name", "serving", "time",
               "ingredient", "spicy_ingredient", "method"]

    # CLOB 포함 row → dict 변환을 DB 연결이 살아 있을 때 해줌
    result = _row_to_dict(columns, row)

    cur.close()     # ← 이제 닫아도 됨
    conn.close()

    return result

def get_categories_by_recipe_id(recipe_id: int) -> List[str]:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT tc.type_name
        FROM recipe_type_category rtc
        JOIN type_category tc
        ON rtc.type_id = tc.type_id
        WHERE rtc.recipe_id = :rid
    """, {"rid": recipe_id})

    rows = cur.fetchall()
    cur.close()
    conn.close()

    # [('국-탕',), ('찌개',)] → ['국-탕', '찌개']
    return [r[0] for r in rows]

def load_all_recipe_categories() -> Dict[int, List[str]]:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT rtc.recipe_id, tc.type_name
        FROM recipe_type_category rtc
        JOIN type_category tc
        ON rtc.type_id = tc.type_id
        ORDER BY rtc.recipe_id
    """)

    rows = cur.fetchall()

    mapping = {}
    for rid, cname in rows:
        mapping.setdefault(rid, []).append(cname)

    cur.close()
    conn.close()
    return mapping

def get_all_recipes() -> List[Dict]:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            recipe_id,
            name,
            serving,
            time,
            ingredient,
            spicy_ingredient,
            method
        FROM recipe
        ORDER BY recipe_id
    """)

    columns = [col[0] for col in cur.description]   # 컬럼 이름 가져오기
    rows = cur.fetchall()

    result = [_row_to_dict(columns, row) for row in rows]

    cur.close()
    conn.close()

    return result

