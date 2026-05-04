from collections import defaultdict
from typing import List, Dict
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import BallTree


def cluster_balltree(
    trees: List["ASTNode"],  # 树列表 # type: ignore
    dis_threshold: int,  # 距离阈值
    encoder: "Encoder",  # type: ignore
):
    # 编码树
    data = np.array([encoder.single_encoding(tree) for tree in trees])

    # 构建 BallTree
    ball_tree = BallTree(data, metric='euclidean')

    # 构建对称距离矩阵
    n = data.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        dist, idx = ball_tree.query(data[i].reshape(1, -1), k=n)
        sorted_dist = dist.flatten()[np.argsort(idx)]  # 按照idx重排序
        dist_matrix[i, :] = sorted_dist.flatten()

    # 进行层次聚类
    clustering = AgglomerativeClustering(
        metric='precomputed', 
        linkage='average', 
        distance_threshold=dis_threshold, 
        n_clusters=None
    )
    clustering.fit(dist_matrix)

    # 构造聚类结果
    clusters = defaultdict(list)
    for i, label in enumerate(clustering.labels_):
        clusters[label].append(trees[i])

    return dict(clusters)



