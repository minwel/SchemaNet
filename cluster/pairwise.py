from typing import Dict, List
from typing import List, Dict


def cluster_matrix_pdist(
    trees: List["ASTNode"],  # 树列表 # type: ignore
    dis_threshold: int,  # 距离阈值
    encoder: "Encoder",  # type: ignore
    encoder_map: Dict[int, str],
    tc_num_fuzzed: int,
    executor: "Executor" = None,  # type: ignore
    test_fuzzer: "TestFuzzer" = None,  # type: ignore
    disable_taskbar: bool = False
) -> Dict[int, List["ASTNode"]]:  # type: ignore
    """
    基于距离矩阵的层次聚类，距离矩阵使用pdist计算，无需数据点可表示为向量。
    但需要自定义距离计算函数，该函数默认为io，暂时写死了。

    :param trees: 待聚类树列表
    :param dis_threshold: 距离阈值
    :param encoder: 编码器
    :param executor: 执行器
    :param test_fuzzer: 测试器
    :param disable_taskbar: 是否禁用进度条
    :return: 聚类结果
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    data = [[x] for x in trees if x.is_correct]
    def tmp_func(x, y, **kwargs):
        dis_res = x[0].get_dis(
            value=y[0],
            encoder=encoder,
            executor=executor,
            test_fuzzer=test_fuzzer,
            tc_num_fuzzed=tc_num_fuzzed,
            encoder_map=encoder_map
        )
        return dis_res.dis

    if len(data) == 1:
        data.append([data[0][0]])
    distance_matrix = pdist(
        data,
        metric=tmp_func,
        taskbar_str="clustering matrix",
        disable=disable_taskbar,
    )

    Z = linkage(distance_matrix, method='average')  # 使用平均链接法
    labels = fcluster(Z, t=dis_threshold, criterion='distance')  # 聚类结果
    # 聚类结果
    cls_dict = {}
    for label, tree in zip(labels, trees):
        cls_dict.setdefault(label, []).append(tree)
    return cls_dict

