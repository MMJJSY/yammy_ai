from sentence_transformers import SentenceTransformer
import numpy as np

# KorSBERT 로딩
model = SentenceTransformer("jhgan/ko-sroberta-multitask")

def get_embedding(text: str) -> np.ndarray:
    """
    문장 하나를 korSBERT 임베딩(768차원)으로 변환
    """
    if not isinstance(text, str):
        text = str(text)

    embedding = model.encode(text)
    return embedding

