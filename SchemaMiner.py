import os
import sys
sys.path.append(os.path.dirname(__file__))
import pickle
import time
import numpy as np
from sklearn.neighbors import BallTree
from encoder.cent import CentEncoder
from encoder.io_encoder import IOEncoder
from encoder.test_gen import TestFuzzer
from trees.tree import Trees
from utils.file import check_paths
from utils.other import (
    ConfigManager,
    FIXED_CLUSTER_FUNC_1,
    FIXED_CLUSTER_FUNC_2,
    FIXED_ENCODE_ALG_1,
    FIXED_ENCODE_ALG_2,
)
from cluster.cluster_strategy import *
from utils.other import print_split_line
from containers import Application
from dependency_injector.wiring import Provide, inject
from utils.executor import Executor
from utils.codedataloader import CodeDataLoader
from typing import Dict


@inject
def init_paths(
    config_manager: ConfigManager = Provide[Application.config_manager],
):
    check_paths(
        config_manager.json_output_path,  # 动态变量变化值信息存储位置 out/trace_info_json/code_from/
        os.path.dirname(config_manager.static_dict_path), # 第1阶段聚类结果保存位置 tmp/cluster_strategy/code_from/static/
        os.path.dirname(config_manager.cls_tree_path)  # 第2阶段聚类结果保存位置 tmp/cluster_strtegy/code_from/dynamic/
    )

@inject
def load_code_data(
    dataloader: CodeDataLoader = Provide[Application.dataloader],
    config_manager: ConfigManager = Provide[Application.config_manager],
    trees: Trees = Provide[Application.trees],
    test_trees: Trees = Provide[Application.test_trees],
):
    """ 加载学生代码，并进行子树分割，获取学生代码中的所有子树 """
    config = config_manager.config
    
    print(f'code from: {config["code_from"]}')
    print(f"code src: {config['code_src']}")

    train_fnames, test_fnames, train_codes, test_codes = dataloader.get4cluster()
    train_set = list(zip(train_fnames, train_codes))
    test_set = list(zip(test_fnames, test_codes))

    # trees数据加载，分块+解析为ASTNode
    trees.add_batch(train_set, True) 
    test_trees.add_batch(test_set, True)

    print(f"train code_num: {len(train_set)}, tree_num: {len(trees)}")
    print(f"test code num: {len(test_set)}, tree_num: {len(test_trees)}")

@inject
def encode_data(
    config_manager: ConfigManager = Provide[Application.config_manager],
    cent_encoder: CentEncoder = Provide[Application.encoder_cent],
    io_encoder: "IOEncoder" = Provide[Application.encoder_io],
):
    """ 对子树进行编码 """
    print_split_line("encoding")

    cent_encoder.encode()  # 第一层编码plan
    io_encoder.encode()  # 第二层编码goal

@inject
def cluster_stage1(
    config_manager: ConfigManager = Provide[Application.config_manager],
    strategy: ClusterStrategy = Provide[Application.strategy],
    trees: Trees = Provide[Application.trees],
    executor: Executor = Provide[Application.executor],
    test_fuzzer: TestFuzzer = Provide[Application.test_fuzzer],
):
    """Step 1: 静态 plan 聚类"""
    config = config_manager.config
    static_dict_path = config_manager.static_dict_path

    print(f"1、一次聚类plan, use encode_alg: {FIXED_ENCODE_ALG_1}")
    print(f"cluster func: {FIXED_CLUSTER_FUNC_1}")
    st_time = time.time()

    force_recluster = bool(config.get("force_recluster", False))

    if force_recluster or not os.path.exists(static_dict_path):
        static_dict = strategy.cluster(
            trees=trees.trees,
            encoder_type=FIXED_ENCODE_ALG_1,
            dis_threshold=config_manager.dis_threshold[FIXED_ENCODE_ALG_1],
            type=1,
            executor=executor,
            test_fuzzer=test_fuzzer,
            disable_taskbar=False
        )
        with open(static_dict_path, 'wb') as f:
            pickle.dump(static_dict, f)
        print(f"static_dict saved to {static_dict_path}")
    else:
        print(f"直接读取static_dict from: {static_dict_path}")
        with open(static_dict_path, 'rb') as f:
            static_dict = pickle.load(f)

    static_dict = sorted(static_dict.items(), key=lambda x: len(x[1]), reverse=True)
    plan_cnt_limit = len(static_dict[0][1]) / config['cls_cnt_limit']
    print(f"plan_cnt_limit: {plan_cnt_limit}")

    static_dict = {
        plan: _trees for plan, _trees in static_dict
        if len(_trees) >= plan_cnt_limit
    }
    end_time = time.time()
    print(f"一次聚类plan耗时: {end_time - st_time} s")
    print(f"有效plans: {len(static_dict)} 个")
    
    sbs_sum = sum([len(plans) for plans in static_dict.values()])
    print(f"有效聚类子树数量: {sbs_sum} 个")

    return static_dict

@inject
def cluster_stage2(
    static_dict: Dict,
    config_manager: ConfigManager = Provide[Application.config_manager],
    strategy_gp: ClusterStrategy = Provide[Application.strategy_gp],
    executor: Executor = Provide[Application.executor],
    test_fuzzer: TestFuzzer = Provide[Application.test_fuzzer],
):
    """Step 2: 动态聚类"""
    print(f"2、二次聚类, use encode_alg: {FIXED_ENCODE_ALG_2}")
    print(f"cluster func: {FIXED_CLUSTER_FUNC_2}")
    st_time = time.time()

    cls_tree = strategy_gp.cluster(
        trees=static_dict,
        encoder_type=FIXED_ENCODE_ALG_2,
        dis_threshold=config_manager.dis_threshold[FIXED_ENCODE_ALG_2],
        type=2,
        executor=executor,
        test_fuzzer=test_fuzzer,
        disable_taskbar=False
    )

    end_time = time.time()
    print(f"二次聚类goal耗时: {end_time - st_time} s")
    print(f"有效goals: {len(cls_tree)} 个")

    # 缓存 sb2cls 映射
    sb2cls = {
        tree.subtree_id: (_cls, _type)
        for _cls, static_dict in cls_tree.items()
        for _type, _trees in static_dict.items()
        for tree in _trees
    }

    with open(config_manager.ast_dict_path, 'wb') as f:
        pickle.dump(sb2cls, f)

    return cls_tree

@inject
def get_encodes_onehot(
    plans: list,  # List[ASTNode], 所有的plans
    bug_tree: "ASTNode",  # type: ignore
    config_manager: ConfigManager = Provide[Application.config_manager],
    encoder_cent: CentEncoder = Provide[Application.encoder_cent]
):
    
    encoder = encoder_cent
    # 获取bug_code的所有子树的编码，默认使用cent
    bug_sbs = bug_tree.get_sub_trees(
        {}
    )
    bug_encodes, _ = encoder.encode_cls_batch(bug_sbs)
    
    # 构建plans的BallTree
    data = np.array([encoder.single_encoding(sb) for sb in plans])
    ball_tree = BallTree(data, metric='euclidean')
    
    # 保持cent维度和ball tree一致
    bug_encodes = [arr[:len(data[0])] for arr in bug_encodes]
    
    # 使用BallTree加速plan匹配
    bug_plan_encodes = [0] * len(plans)  # plan one-hot 编码
    bug_encoded_plans = []  # bug sbs分别匹配的plan，若没匹配上则为None
    for sb_idx in range(len(bug_sbs)):
        bug_sb_encoding = bug_encodes[sb_idx].reshape((1, -1))
        dis, matched_plan_idx = map(lambda x: x.item(), ball_tree.query(
            bug_sb_encoding,
            k = 1
        ))

        bug_encoded_plans.append(plans[matched_plan_idx])
        bug_plan_encodes[matched_plan_idx] = 1
        

    return bug_plan_encodes

def get_clustered_res(
    is_plan: bool = False,
    config_manager: ConfigManager = Provide[Application.config_manager],
    encoder_cent: CentEncoder = Provide[Application.encoder_cent],
):

    # 加载cls_tree
    cls_tree_path = config_manager.cls_tree_path
    if os.path.exists(cls_tree_path):
        print(f"直接读取cls_tree from: {cls_tree_path}")
        with open(cls_tree_path, 'rb') as f:
            cls_tree = pickle.load(f)
        if is_plan:
            load_code_data()  # 加载trees代码
            encoder_cent.encode()  # 编码
    else:
        # 初始化路径
        init_paths()

        # 加载trees代码
        load_code_data()  

        # 代码编码
        encode_data()

        # 代码聚类
        static_dict = cluster_stage1()
        cls_tree = cluster_stage2(static_dict)
        
        # 保存结果
        with open(cls_tree_path, 'wb') as f:
            pickle.dump(cls_tree, f)
        print(f"cls_tree saved to {cls_tree_path}")

    return cls_tree
