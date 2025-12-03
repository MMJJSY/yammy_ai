import oracledb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# 1) Oracle 연결 준비
# ─────────────────────────────────────────────
oracledb.init_oracle_client(lib_dir=r"C:\oracle\instantclient_19_29")

conn = oracledb.connect(
    user="yammy",
    password="1234",
    dsn=oracledb.makedsn("localhost", 1521, service_name="XEPDB1")
)

cur = conn.cursor()

# ─────────────────────────────────────────────
# 2) DB에서 모든 레시피 가져오기
# ─────────────────────────────────────────────
cur.execute("""
    SELECT recipe_id, name, ingredient, spicy_ingredient
    FROM recipe
    ORDER BY recipe_id
""")

rows = cur.fetchall()

# 각 레시피를 dict로 변환
recipes = []
for r in rows:
    recipes.append({
        "id": r[0],
        "name": r[1],
        "ingredient": r[2],
        "spicy_ingredient": r[3]
    })

print(f"불러온 레시피 개수: {len(recipes)}")

# ─────────────────────────────────────────────
# 3) SBERT 모델 로드
# ─────────────────────────────────────────────
model = SentenceTransformer("jhgan/ko-sroberta-multitask")

# 제목 + 재료 + 양념을 모두 합쳐 임베딩 텍스트로 사용
texts = [
    f"{r['name']} {r.get('ingredient', '')} {r.get('spicy_ingredient', '')}"
    for r in recipes
]

# 임베딩 생성
vectors = model.encode(texts)
vectors = np.array(vectors)

print("임베딩 완료 → shape:", vectors.shape)

# ─────────────────────────────────────────────
# 4) 검색 함수 정의
# ─────────────────────────────────────────────
def search(query):
    q_vec = model.encode([query])
    sims = cosine_similarity(q_vec, vectors)[0]
    idx = np.argmax(sims)
    return recipes[idx], sims[idx]

# ─────────────────────────────────────────────
# 5) 테스트 실행
# ─────────────────────────────────────────────
query = "칼칼한 돼지고기 찌개 먹고싶어"

result, score = search(query)

print(f"\n[검색어] {query}")
print(f"[추천 레시피] {result['name']}")
print(f"[유사도] {score:.4f}")