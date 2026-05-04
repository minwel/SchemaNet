import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from typing import Dict, List
from contextlib import contextmanager

FIXED_CLUSTER_STRATEGY = "pairwise"
FIXED_CLUSTER_FUNC_1 = "ball"
FIXED_CLUSTER_FUNC_2 = "matrix"
FIXED_ENCODE_ALG_1 = "cent"
FIXED_ENCODE_ALG_2 = "io"

class ConfigManager:
    def __init__(self, config):
        self.cluster_strategy = FIXED_CLUSTER_STRATEGY
        self.cluster_func_1 = FIXED_CLUSTER_FUNC_1
        self.cluster_func_2 = FIXED_CLUSTER_FUNC_2
        self.encode_alg_1 = FIXED_ENCODE_ALG_1
        self.encode_alg_2 = FIXED_ENCODE_ALG_2

        self._resolve_dataset_paths(config)

        self.config = config
        # 源路径
        # 聚类代码的来源（原代码路径，包含所有原代码文件的文件夹）
        self.code_path = os.path.join(
            config["root_path"],
            config["code_src"]
        )
        # 实际待聚类的代码（以excel文件存在）
        self.true_code_path = os.path.join(
            config["root_path"],
            config["true_code_path"]
        )


        # 保存路径
        # 第1阶段聚类结果保存位置
        static_clustered_dict = (
            f"cls_dict_{self.encode_alg_1}_"
            f"{self.cluster_func_1}.pkl"
        )
        self.static_dict_path = os.path.join(
            config["root_path"],
            config["tmp_path"],
            self.cluster_strategy,
            config["code_from"],
            "static",
            static_clustered_dict
        )
        # 第2阶段聚类结果保存位置
        cls_tree_dict = (
            f"cls_dict_{self.encode_alg_1}_"
            f"{self.encode_alg_2}_{self.cluster_func_1}.pkl"
        )
        self.cls_tree_path = os.path.join(
            config["root_path"],
            config["tmp_path"],
            self.cluster_strategy,
            config["code_from"],
            "dynamic",
            cls_tree_dict
        )
        self.ast_dict_path = os.path.join(
            config["root_path"],
            config["tmp_path"],
            self.cluster_strategy,
            config["code_from"],
            config["ast_dict_pkl_fname"]
        )
        # 动态变量变化值信息存储位置
        self.json_output_path = os.path.join(
            config["root_path"],
            "out/trace_info_json",
            config["code_from"],
        )
        # 超参数
        self.dis_threshold = config["dis_threshold"]
        
        self.encoder_map = {
            1: self.encode_alg_1,
            2: self.encode_alg_2
        }


    @staticmethod
    def _resolve_dataset_paths(config: Dict):
        code_from = str(config.get("code_from", "")).strip()

        code_src = config.get("code_src", "").strip()
        if not code_src:
            code_src = f"data/test/{code_from}"
        elif "{code_from}" in code_src:
            code_src = code_src.format(code_from=code_from)

        true_code_path = config.get("true_code_path", "").strip()
        if not true_code_path:
            true_code_path = f"data/test/{code_from}.xlsx"
        elif "{code_from}" in true_code_path:
            true_code_path = true_code_path.format(code_from=code_from)

        code_src = ConfigManager._dedupe_trailing_segment(code_src, code_from)
        true_code_path = ConfigManager._dedupe_true_code_path(true_code_path, code_from)

        config["code_src"] = code_src
        config["true_code_path"] = true_code_path

    @staticmethod
    def _dedupe_trailing_segment(path_str: str, code_from: str) -> str:
        parts = list(Path(path_str).parts)
        if len(parts) >= 2 and parts[-1] == parts[-2] and parts[-1] == code_from:
            return str(Path(*parts[:-1]))
        return path_str

    @staticmethod
    def _dedupe_true_code_path(path_str: str, code_from: str) -> str:
        p = Path(path_str)
        if p.stem == code_from and p.parent.name == code_from:
            return str(p.parent.parent / f"{code_from}{p.suffix}")
        return path_str

def prepare_csv_field(field):
    # 转换字段为字符串，防止非字符串类型引发错误
    field = str(field)
    # 如果字段包含逗号或者双引号，则包裹在双引号内
    if ',' in field or '"' in field:
        # 先将字段中的双引号替换为两个双引号
        field = field.replace('"', '""')
        # 包裹双引号
    return f'"{field}"'

def save_goal_res(
    output_path: str, 
    cls_tree: Dict[int, Dict[int, List["ASTNode"]]],  # type: ignore
):
    """ 
    保存结果，共6个文件

    params
    ------
    cls_tree : 经过过滤的聚类结果（去掉了小于3个子树的类），以及进行了plan的聚类
    ...
    """
    print_split_line('saveing res')
    output_path = output_path +"/goals"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # 1、聚类结果统计表, count_cls.csv
    count_cls_out = open(os.path.join(output_path, 'goal_info.csv'), "w", encoding="utf8")
    count_cls_out.write('Goal_id,Goal_count,subtree_text\n')
    # 2、聚类结果详细表, count.csv
    count_out = open(os.path.join(output_path, 'count.csv'), "w", encoding="utf8")
    count_out.write(
        "cls,cls_1,subtree_id,subtree_txt,cls_dis,cls_dis1,"
        "vars,var_map,is_correct,code_id,fname\n"
    )

    hist_data = []  # 直方图数据
    for _cls, _cls_dict in cls_tree.items():
        cls_count = sum([len(x) for x in _cls_dict.values()])
        hist_data += [_cls] * cls_count  # 记录直方图数据

        # 写count_cls.csv
        tmp = []
        tmp.append(str(_cls))  # cls
        tmp.append(str(cls_count))  # cls_count
        # 获取该goal下最多数量的plan的第一个树作为代表
        repr_tree = _cls_dict[next(iter(_cls_dict))][0]
        tmp.append(prepare_csv_field(repr_tree.get_text()))
        count_cls_out.write(','.join(tmp) + '\n')
        
        # 写count.csv，只会展示大于cnt_limit的分类的树
        for sb in (sb for _sbs in _cls_dict.values() for sb in _sbs):  # 生成器遍历所有树
            sb: "ASTNode"  # type: ignore
            tmp = []
            # tmp.append(str(sb.type_1))  # cls
            tmp.append(str(_cls))  # cls，因为goal的聚类可能是从pkl加载的，所有type_1都是-1
            tmp.append(str(sb.type_2))  # cls_1
            tmp.append(str(sb.subtree_id))  # subtree_id
            tmp.append(prepare_csv_field(sb))  # subtree_txt
            tmp.append(prepare_csv_field(sb.dis_res_1))  # cls_dis
            tmp.append(prepare_csv_field(sb.dis_res_2))  # cls_dis1

            tmp.append(prepare_csv_field(sb.vars))  # vars
            tmp.append(prepare_csv_field(sb.dis_res_1._map))  # var_map
            tmp.append(str(sb.is_correct))  # is_correct
            tmp.append(str(sb.code_id))  # code_id
            tmp.append(sb.f_name)  # fname
            count_out.write(','.join(tmp) + '\n')
    count_cls_out.close()
    count_out.close()

    # 4、goal_plan_info表，gap_tree.csv
    count_tree_out = open(os.path.join(output_path, f'goal_plan_info_{len(cls_tree)}.csv'), "w", encoding="utf8")
    count_tree_out.write(
        "goal_id,plan_id,goal_count,sb_text,sb_id\n"
    )
    count_tree_out.close()


    print(f'cluster results saved to {output_path}')
    return


def save_plan_res(
    output_path: str, 
    cls_tree: Dict[int, List["ASTNode"]],  # type: ignore
):
    # gap_tree表，gap_tree.csv
    output_path = output_path +"/plans"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    count_tree_out = open(os.path.join(output_path, f'plan_info_{len(cls_tree)}.csv'), "w", encoding="utf8")
    count_tree_out.write(
        "plan_id,plan_count,sb_text\n"
    )
    for _cls, _trees in cls_tree.items():
        tmp = []
        tmp.append(str(_cls))  # plan_id
        tmp.append(str(len(_trees)))  # plan_count
        tmp.append(prepare_csv_field(_trees[0].get_text()))  # sb_text
        count_tree_out.write(','.join(tmp) + '\n')
        
    count_tree_out.close()

@contextmanager
def cal_time(name):
    """ 计算操作用时 """
    import time
    start = time.time()
    yield
    print(f"{name} took {(time.time() - start) / 60 :.4f} Mins")


def print_split_line(title: str, width: int=60):
    offset = int(width / 2 - len(title) / 2)
    print(f"{'-' * offset} {title} {'-' * offset}")
