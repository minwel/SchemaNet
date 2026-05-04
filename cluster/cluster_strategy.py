import sys
from pathlib import Path

from cluster.cluster_ball import cluster_balltree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from overrides import override
from pairwise import cluster_matrix_pdist
from abc import ABC, abstractmethod
from typing import List
from utils.other import FIXED_ENCODE_ALG_1, FIXED_ENCODE_ALG_2

class ClusterStrategy(ABC):
    @abstractmethod
    def cluster(
        self,
        trees,  # type: ignore
        encoder_type: str,
        dis_threshold: float,
        **kwargs,
    ) -> dict:
        ...

class Pairwise(ClusterStrategy):
    def __init__(self, config_manager, **kwargs) -> None:
        self.config = config_manager.config
        self.encoder_map = config_manager.encoder_map
        self.encoders = kwargs

    @override
    def cluster(
        self,
        trees: List["ASTNode"],  # type: ignore
        encoder_type: str,
        dis_threshold: float,
        **kwargs,
    ) -> dict:
        """ 
        encoder_type: "io", "cent"
        """

        # 固定聚类函数：plan -> ball, goal -> matrix
        cluster_func = cluster_balltree if kwargs["type"] == 1 else cluster_matrix_pdist

        # 聚类，返回类别dict
        cls_dict = cluster_func(
            trees=trees,
            dis_threshold=dis_threshold,
            encoder=self.encoders[encoder_type],
        )
        
        # 用聚类结果cls_dict更新sb的类别属性
        for _cls, _trees in cls_dict.items():
            if len(_trees) == 1:
                continue
            for tree in _trees:             
                setattr(
                    tree,
                    {
                        FIXED_ENCODE_ALG_1: "type_1",
                        FIXED_ENCODE_ALG_2: "type_2",
                    }.get(encoder_type, "type_1"),
                    _cls,
                )

        return cls_dict

class Pairwise_gp(ClusterStrategy):
    def __init__(self, config_manager, **kwargs) -> None:
        self.config = config_manager.config
        self.encoder_map = config_manager.encoder_map
        self.encoders = kwargs

    @override
    def cluster(
        self,
        trees,  # type: ignore
        encoder_type: str,
        dis_threshold: float,
        **kwargs,
    ) -> dict:
        """ 
        encoder_type: "io", "cent", "edit"
        此处trees实为cls_dict
        """
        # 固定聚类函数：plan -> ball, goal -> matrix
        cluster_func = cluster_balltree if kwargs["type"] == 1 else cluster_matrix_pdist
        tmp_trees = [ts[0] for k, ts in trees.items()]
        tmp_dict = {ts[0].subtree_id: ts for k, ts in trees.items()}
            
        cls_dict = cluster_func(
            trees=tmp_trees,
            dis_threshold=dis_threshold,
            encoder=self.encoders[encoder_type],
        )

        cls_dict_res = {}
        for _cls, _trees in cls_dict.items():
            for idx, tree in enumerate(_trees):
                cls_dict_res.setdefault(_cls, {})[idx + 1] = tmp_dict[tree.subtree_id]

        # 用聚类结果cls_dict_res更新sb的goal类别，即type_2
        for goal_idx, plans in cls_dict_res.items():
            for plan_idx, trees in plans.items():
                for tree in trees:                    
                    setattr(
                        tree,
                        {
                            FIXED_ENCODE_ALG_1: "type_1",
                            FIXED_ENCODE_ALG_2: "type_2",
                        }.get(encoder_type, "type_1"),
                        goal_idx,
                    )

        return cls_dict_res