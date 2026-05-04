import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from utils.str_type import PathString
from utils.tracer import Tracer
import numpy as np
from collections import defaultdict
from static.structs import LineTraceInfo
from utils.dataloader import TestCasesLoader
from utils.executor import Executor
from utils.other import ConfigManager
from utils.file import read_yaml


# init tracer 
config = read_yaml("config.yml")
test_cases = TestCasesLoader.get_test_cases("data/testcase")
executor = Executor(test_cases, ConfigManager(config))
tracer = Tracer(
    executor=executor, config_manager=ConfigManager(config)
)

def extract_start_end_values(trace_res):
    """
    从 TraceRes 提取每个变量在 5 个测试用例下的初始值 & 结束值

    :param trace_res: TraceRes 对象
    :return: {'var1': {'start': [val1, val2, ...], 'end': [val1, val2, ...]}, ...}
    """
    # 使用 defaultdict 来存储变量的初始值和结束值
    var_traces = defaultdict(lambda: {"start": [None] * trace_res.tc_num, "end": [None] * trace_res.tc_num})

    # 遍历 trace_res 中的所有 trace
    for line_no, traces in trace_res.lines_trace_info.items():
        for tc_idx, trace in enumerate(traces):
            if isinstance(trace, LineTraceInfo):  # 确保 trace 是 LineTraceInfo 类型
                # 遍历当前 trace 中的 start 和 end 值
                for var_name, val in trace.start.items():
                    if var_traces[var_name]["start"][tc_idx] is None:  # 如果当前 test case 中没有赋值，说明是首次出现
                        var_traces[var_name]["start"][tc_idx] = val
                for var_name, val in trace.end.items():
                    var_traces[var_name]["end"][tc_idx] = val  # 始终更新最后一次出现的 end 值
    
    return var_traces

def encode_variable_features(var_traces):
    """
    将提取的变量数据编码为数值特征
    :param var_traces: 提取的变量值字典
    :return: 归一化后的特征向量
    """
    type_encodings = {'int': 0, 'float': 1, 'str': 2, 'list': 3, 'tuple': 4}
    all_features = []

    for var_name, values in var_traces.items():
        start_values = values["start"]
        end_values = values["end"]

        # 变量类型 one-hot
        first_val = next((v for v in start_values if v is not None), None)
        var_type = type(first_val).__name__ if first_val is not None else "unknown"
        type_vector = np.zeros(len(type_encodings))
        if var_type in type_encodings:
            type_vector[type_encodings[var_type]] = 1

        # 计算统计特征
        stats_features = []

        for val_set in [start_values, end_values]:
            filtered_values = [v for v in val_set if v is not None]  # 过滤 None

            if filtered_values:  # 如果过滤后的列表非空
                if all(isinstance(v, (int, float)) for v in filtered_values):  # 确保所有值是数值
                    num_arr = np.array(filtered_values, dtype=np.float32)
                    stats_features.extend([np.mean(num_arr), np.std(num_arr), np.max(num_arr), np.min(num_arr)])

                elif all(isinstance(v, str) for v in filtered_values):  # 确保所有值是字符串
                    str_lengths = np.array([len(v) for v in filtered_values], dtype=np.float32)
                    stats_features.extend([np.mean(str_lengths), np.std(str_lengths), len(set(filtered_values))])

                elif all(isinstance(v, (list, tuple)) for v in filtered_values):  # 确保所有值是 list/tuple
                    list_lengths = np.array([len(v) for v in filtered_values], dtype=np.float32)
                    stats_features.extend([np.mean(list_lengths), np.std(list_lengths), len(filtered_values)])
            else:
                # 如果没有有效的值，可以选择填充默认值，如零值或跳过特征计算
                stats_features.extend([0, 0, 0, 0])  # 或者选择其他默认值

        # 归一化
        stats_features = np.array(stats_features)
        if len(stats_features) > 0:
            stats_features = (stats_features - np.mean(stats_features)) / (np.std(stats_features) + 1e-6)

        # 拼接最终特征
        final_feature_vector = np.concatenate([type_vector, stats_features])
        all_features.append(final_feature_vector)

    return np.array(all_features, dtype=object)


def get_dynamics(code_path):
    trace_res = tracer.trace_execution(PathString(code_path))

    # 提取变量初始值和结束值
    extracted_vars = extract_start_end_values(trace_res)

    # 变量编码
    encoded_features = encode_variable_features(extracted_vars)

    
    if len(encoded_features) == 0:
        return np.array([0] * 64)


    # 找出最长的特征向量长度
    max_len = max(len(vec) for vec in encoded_features)
    # 补零到相同长度
    padded_features = [np.pad(vec, (0, max_len - len(vec)), 'constant') for vec in encoded_features]
    encoded_features = np.vstack(padded_features)  # 变成 2D 矩阵
    return encoded_features.flatten()  # 再展平成 1D

