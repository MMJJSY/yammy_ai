from fastapi import FastAPI
from app.apis.analyze import router as analyze_router
from app.apis.recommend import router as recommend_router

app = FastAPI()

# /api/analyze
app.include_router(analyze_router, prefix="/api")

# /api/recommend
app.include_router(recommend_router, prefix="/api")

@app.get("/")
def root():
    return {"message": "Yammy API Running!"}