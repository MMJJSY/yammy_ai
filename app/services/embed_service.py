from sentence_transformers import SentenceTransformer
import numpy as np

# SBERT 모델 로딩 (서버 시작 시 한 번만)
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

def get_embedding(text: str) -> np.ndarray:
    """
    문장 하나를 KR-SBERT 임베딩(768차원)으로 변환
    """
    if not isinstance(text, str):
        text = str(text)

    embedding = model.encode(text)
    return embedding