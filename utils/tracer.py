import pickle
import os
import sys
sys.path.append(os.path.dirname(__file__))
from static.structs import LineTraceInfo, TraceRes
from utils.other import ConfigManager
from utils.file import check_paths
from utils.str_type import *

class Tracer:
    def __init__(
        self,
        executor: "Executor",
        config_manager: ConfigManager,
    ):
        self.config = config_manager.config
        self.json_output_path = config_manager.json_output_path
        self.executor = executor

    def trace_execution(self, f_path: PathString) -> TraceRes:
        """
        跟踪代码执行轨迹，输入为代码路径。
        自动缓存 trace 结果，避免重复执行。
        """
        res_pickle_path = PathString(self.config["root_path"]).join(
            self.json_output_path, f_path.get_filename_without_ext() + ".pkl"
        )
        check_paths(os.path.dirname(res_pickle_path))

        # 已缓存
        if os.path.exists(res_pickle_path):
            with open(res_pickle_path, "rb") as f:
                return pickle.load(f)

        # 未缓存
        code_str = CodeString(f_path.read())
        exec_res = self.executor.execute_code(
            code_str=code_str,
            prob_id=self.config["code_from"],
            enable_trace=True,
        )
        
        line_trace_map = {line: [] for line in range(1, code_str.get_code_line_count() + 1)}
        total_test_cases = 0

        for test_case_idx, test_case_trace in enumerate(exec_res.trace_res):  # 每个测试用例的执行结果
            if test_case_trace  == 0:
                # 执行失败或超时，补 None
                for traces in line_trace_map.values():
                    traces.append(None)
                continue

            prev_trace_len = len(line_trace_map[1]) # 记录当前测试用例编号

            for line, trace_info in test_case_trace.items():
                line_trace_map.setdefault(line, []).append(
                    LineTraceInfo.from_dict(trace_info)
                )

            # 补充未覆盖行的 None
            for line in line_trace_map:
                if len(line_trace_map[line]) == prev_trace_len:
                    line_trace_map[line].append(None)
            
            # 保证所有行都补全了当前 test case 的 trace
            assert all(len(traces) == prev_trace_len + 1 for traces in line_trace_map.values()), \
                "测试用例 trace 补全失败"

            total_test_cases += 1

        trace_result = TraceRes(
            lines_trace_info=line_trace_map,
            lines_seq=exec_res.trace_lines_seq,
            tc_num=total_test_cases
        )
        # 缓存trace结果，以便下次读取
        with open(res_pickle_path, 'wb') as f:
            pickle.dump(trace_result, f)
        # print(trace_result)
        return trace_result
