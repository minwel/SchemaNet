import numpy as np

def get_cosine_dis(x1: np.ndarray, x2: np.ndarray):
    # 计算向量的点积
    dot_product = np.dot(x1, x2.T)

    # 计算向量的范数（模）
    norm_a = np.linalg.norm(x1)
    norm_b = np.linalg.norm(x2)

    # 检查零向量
    if norm_a == 0 or norm_b == 0:
        raise ValueError("One of the input vectors is a zero vector, cosine similarity is undefined.")

    # 计算余弦相似度，并限制范围
    cosine_similarity = dot_product / (norm_a * norm_b)
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)

    # 计算余弦距离，进行非负映射
    adjusted_similarity = (cosine_similarity + 1) / 2
    cosine_distance = 1 - adjusted_similarity

    return cosine_distance

