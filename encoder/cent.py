import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from overrides import override
from tqdm import tqdm
import numpy as np
import networkx as nx
from typing import List
from base import SubtreeEncoder


class CentEncoder(SubtreeEncoder):
    def __init__(
        self,
        trees: "Trees",  # type: ignore
        *args,
        **kwargs
    ) -> None:
        super().__init__(trees, *args, **kwargs)
        self.trees = trees
        self.g = nx.Graph()
        self.sbs_vecs = {}
        self.type2index = {}

    def encode(self):
        """ 初始化时对所有树编码 """
        # 获取所有树的节点type，并编号
        all_types = set()
        for sb in self.trees.trees:
            all_types.update(sb.get_all_node_type())

        self.type2index = {tp: index for index, tp in enumerate(all_types)}

        # 给图加入所有树的所有节点
        for tp, index in self.type2index.items():
            self.g.add_node(index, name = tp)

        # 对每棵树进行编码
        for sb in tqdm(self.trees.trees, total=len(self.trees.trees), desc="cent encoding"):
            # 仅清除边，保证节点数量相等，每棵树的嵌入维度相等
            self.g.clear_edges()
            cent_matrix = self.get_single_cent_matrix(sb)
            self.sbs_vecs[sb.subtree_id] = cent_matrix

        print(f"encoded dims: {len(next(iter(self.sbs_vecs.values())))}")

    @override
    def single_encoding(self, subtree: "ASTNode") -> np.ndarray:
        if subtree.subtree_id in self.sbs_vecs:
            return self.sbs_vecs[subtree.subtree_id]
        else:
            raise ValueError("subtree not in sbs_vecs")
    
    def update_encode(self, ):
        # 对每棵树进行编码
        for sb in tqdm(self.trees.trees, total=len(self.trees.trees), desc="update encoding"):
            # 仅清除边，保证节点数量相等，每棵树的嵌入维度相等
            self.g.clear_edges()
            cent_matrix = self.get_single_cent_matrix(sb)
            self.sbs_vecs[sb.subtree_id] = cent_matrix
    
    def encode_cls_batch(self, subtrees: List["ASTNode"]) -> np.ndarray:
        """ 批量编码，且可能含有新的节点类型要更新 """
        # 要加入的树可能有新的节点类型，更新之前的树的所有编码和节点编号
        node_types = set()
        for subtree in subtrees:
            node_types.update(subtree.get_all_node_type())
        need_update = False
        for node_type in node_types:
            if not node_type in self.type2index:
                need_update = True                
                self.type2index[node_type] = len(self.type2index)
                self.g.add_node(len(self.type2index), name=node_type)
        if need_update:
            print("该code有新节点加入，需要更新所有cent编码...")
            self.update_encode()
        # 编码新的树
        sbs_vecs = []
        for subtree in subtrees:
            self.g.clear_edges()
            sbs_vecs.append(self.get_single_cent_matrix(subtree))

        return sbs_vecs, need_update
        
    @override
    def batch_encoding(self) -> List[np.ndarray]:
        return list(self.sbs_vecs.values())


    def get_single_cent_matrix(self, tree_root: "ASTNode") -> np.ndarray:
        # 获取所有边
        src = []
        tgt = []
        tree_root.get_node_n_edge(tree_root.ast_node, src, tgt)

        # 向图中加入所有边
        for i in range(len(src)):
            m = self.type2index[src[i]]
            n = self.type2index[tgt[i]]
            if self.g.has_edge(m, n):
                self.g[m][n]['weight'] += 1
            else:
                self.g.add_edge(m, n, weight=1)

        this_all_cents = dict()
        this_all_cents["cent_harm"] = [cent /len(self.g) for cent in nx.harmonic_centrality(self.g).values()]
        this_all_cents["cent_eigen"] = [cent for cent in nx.eigenvector_centrality(self.g).values()]
        this_all_cents["cent_close"] = [cent for cent in nx.closeness_centrality(self.g).values()]
        this_all_cents["cent_between"] = [cent for cent in nx.betweenness_centrality(self.g).values()]
        this_all_cents["cent_degree"] = [cent for cent in nx.degree_centrality(self.g).values()]
        this_all_cents["cent_katz"] = [cent for cent in nx.katz_centrality(self.g).values()]

        res = []
        for x in this_all_cents.values():
            res += x
        return np.array(res)

