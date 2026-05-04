# coding: utf-8
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from static.structs import EncodeRes, TestSeed, TraceRes
from trees.visitor import ReturnValueAnalyzer, extract_variables_from_unparsable_code
from tqdm import tqdm
from utils.str_type import CodeString, PathString
from utils.other import ConfigManager
from base import SubtreeEncoder
from typing import Dict, List, Tuple
from utils.tracer import Tracer
from utils.executor import Executor


class IOEncoder(SubtreeEncoder):
    def __init__(
        self, trees: "Trees" = None, *args, **kwargs  # type: ignore
    ) -> None:  # type: ignore
        self.exec_trace : Dict[str, TraceRes] = {}
        self.sbs_io_info = {}
        config_manager : ConfigManager = kwargs["config_manager"]
        self.code_path = config_manager.code_path
        self.trees = trees
        self.executor: Executor = kwargs["executor"]
        self.tracer: Tracer = kwargs["tracer"]

    def get_io_info(
            self, sb: "ASTNode", f_str: CodeString, # type: ignore
            args_trace_info: TraceRes, in_info: List, in_vars: set,
            out_info: set, out_vars: set, out_consts: List
        ):

        input_args = args_trace_info.get_line_inputargs(sb.start_line)
        lines_seq = args_trace_info.lines_seq  

        for tc_idx in range(args_trace_info.tc_num):
            if tc_idx >= len(input_args):
                continue
            if input_args[tc_idx] == None:
                in_info.append(None)
                out_info.append(None)
                continue
            tc_in_info = {}
            for k, v in input_args[tc_idx].start.items():
                if k in sb.vars[0]:  
                    tc_in_info[k] = v
                    in_vars.add(k)  
            in_info.append(tc_in_info)

            max_idx = -1  
            sb_max_line_no = -1  
            sb_max_line_idx = -1  
            for line, trace_info in args_trace_info.lines_trace_info.items():
                if trace_info[tc_idx] and \
                    trace_info[tc_idx].index > max_idx and \
                    sb.start_line <= line <= sb.end_line:
                    max_idx = trace_info[tc_idx].index
            
                if trace_info[tc_idx] and \
                    sb.start_line <= line <= sb.end_line and \
                    line > sb_max_line_no:
                        sb_max_line_no = line
                        sb_max_line_idx = trace_info[tc_idx].index
            
            assert max_idx != -1 and sb_max_line_idx != -1 and \
                sb_max_line_no != -1, \
                "max_idx == -1 or sb_max_line_idx == -1 or " \
                "sb_max_line_no == -1"

            if sb_max_line_idx < len(lines_seq[tc_idx]):
                next_line_no = lines_seq[tc_idx][sb_max_line_idx]
                next_line_info = args_trace_info.lines_trace_info[next_line_no][tc_idx].end
            else:
                next_line_info = {}

            later_lines_no = sorted(set(lines_seq[tc_idx][max_idx: ])) 
            later_code = [
                f_str.get_lines()[line - 1] for line in later_lines_no
            ]

            out_vars_1 = set()
            for var, val in tc_in_info.items():
                if var in sb.vars[1]:
                    out_vars_1.add(var)
                if var in next_line_info and next_line_info[var] != val:
                    out_vars_1.add(var)

            out_vars_2 = set()
            later_vars = set()
            for line in later_code:
                _vars = extract_variables_from_unparsable_code(line)
                later_vars.update(_vars)
            for _var in sb.vars[1]: 
                if _var in later_vars:
                    out_vars_2.add(_var)
            out_vars_3 = set()
            print_return_val = ReturnValueAnalyzer.analyze_code(sb.text)
            out_vars_3.update(print_return_val['vars'])

            _out_vars = out_vars_1 | out_vars_2 | out_vars_3
            out_vars.update(_out_vars)
            tc_out_info = {
                var: next_line_info[var]
                if var in next_line_info else None
                for var in _out_vars
            } 
            out_consts.update(print_return_val["consts"])
            out_info.append(tc_out_info)

    def get_io_info2(self, sb, f_str, args_trace_info, in_info, in_vars, out_info, out_vars, out_consts):
        start_line = sb.start_line if hasattr(sb, 'start_line') else sb
        end_line = sb.end_line if hasattr(sb, 'end_line') else sb
        
        if isinstance(sb, int):
            start_line = end_line = sb
        
        for tc_idx in range(args_trace_info.tc_num):
            if tc_idx >= len(args_trace_info.lines_seq):
                continue
                
            current_seq = args_trace_info.lines_seq[tc_idx]
            
            block_positions = []
            for i, line_num in enumerate(current_seq):
                if start_line <= line_num <= end_line:
                    block_positions.append(i)
            
            if not block_positions:
                in_info.append(None)
                out_info.append(None)
                continue
        
            first_pos = block_positions[0]
            if first_pos > 0:
                prev_line = current_seq[first_pos - 1]
                if prev_line in args_trace_info.lines_trace_info:
                    trace_infos = args_trace_info.lines_trace_info[prev_line]
                    if tc_idx < len(trace_infos) and trace_infos[tc_idx] is not None:
                        input_state = trace_infos[tc_idx].end.copy()
                    else:
                        input_state = {}
                else:
                    input_state = {}
            else:
                first_line = current_seq[first_pos]
                if first_line in args_trace_info.lines_trace_info:
                    trace_infos = args_trace_info.lines_trace_info[first_line]
                    if tc_idx < len(trace_infos) and trace_infos[tc_idx] is not None:
                        input_state = trace_infos[tc_idx].start.copy()
                    else:
                        input_state = {}
                else:
                    input_state = {}
            
            last_pos = block_positions[-1]
            last_line = current_seq[last_pos]
            if last_line in args_trace_info.lines_trace_info:
                trace_infos = args_trace_info.lines_trace_info[last_line]
                if tc_idx < len(trace_infos) and trace_infos[tc_idx] is not None:
                    output_state = trace_infos[tc_idx].end.copy()
                else:
                    output_state = {}
            else:
                output_state = {}
            
            in_info.append(input_state if input_state else None)
            out_info.append(output_state if output_state else None)
            
            if input_state:
                in_vars.update(input_state.keys())
            if output_state:
                out_vars.update(output_state.keys())
                
            for var, value in output_state.items():
                if isinstance(value, (bool, int, float, str, type(None))):
                    if isinstance(value, (bool, int, float)) or value is None:
                        out_consts.add(value)
        
        while len(in_info) < args_trace_info.tc_num:
            in_info.append(None)
        while len(out_info) < args_trace_info.tc_num:
            out_info.append(None)

    def encode(self) -> EncodeRes:
        if self.sbs_io_info:
            print("io encoding: 已进行过，无需再次编码")
            return
        
        for sb in tqdm(
            self.trees.trees, total=len(self.trees.trees), desc="io encoding"
        ):
            sb: "ASTNode"  # type: ignore

            if not sb.is_correct:
                return EncodeRes(0)
            
            # 所属原代码基本信息
            f_name = sb.f_name
            f_path = PathString(os.path.join(
                self.code_path, f_name
            ))
            try:
                f_str = CodeString(f_path.read()) 
            except:
                print("读取代码文件路径出错")
                return
            

            # 1、获取所属代码文件的trace信息：args_trace_info
            if not f_name in self.exec_trace:
                trace_res = self.tracer.trace_execution(f_path)  # 执行变量跟踪
                self.exec_trace[f_name] = trace_res  # cache
            args_trace_info: TraceRes = self.exec_trace[f_name]

            # 2、获取io信息
            in_info = []  # 该sb对所有tc输入变量和值：None、{}、{'a': 1}
            out_info = []  # 该sb对所有tc输出变量和值：None、{'a': 1}
            in_vars = set()
            out_vars = set()  # 该sb的输出变量：{}、{'a', 'b'}
            out_consts = set() # 该sb的输出常量：{}、{False, True}
            self.get_io_info2(
                sb, f_str, args_trace_info, in_info, in_vars,
                out_info, out_vars, out_consts
            )

            # 2-1、去除sb的vars[0]中没有输入值的变量（过滤掉如：下标变量）
            for var in sb.vars[0]:
                if not var in in_vars:
                    while var in sb.vars[0]:
                        sb.vars[0].remove(var)

            sb.func_body = sb.get_func_body(args=sb.vars[0])

            # 3、转换为EncodeRes
            encode_res = EncodeRes(1, in_info, out_info, out_vars, out_consts)
            params = {}  # {'a': TestSeed('a', int, ['1', '2', '3']), ...}
            for tc_vars in in_info:  # 对每个tc
                if tc_vars == None:  
                    # 该tc未覆盖该sb的起始行，则没有输入，为使每个测试用例都有意义
                    # 添加占位符"UNCOVERED"，后面fuzz生成测试用例
                    for param in sb.vars[0]:  # 输入变量
                        # 当首次添加该param且未覆盖时，使用默认类型int
                        params.setdefault(
                            param, TestSeed(param)
                        ).vals.append("UNCOVERED")
                else:
                    # 覆盖
                    for param in tc_vars:
                        val = tc_vars[param]
                        if param in params:
                            params[param]._type = type(val).__name__
                        else:
                            params[param] = TestSeed(param, type(val).__name__)
                        params[param].vals.append(val)

            encode_res.params = params  # 后续用于fuzz和获取独立io
            encode_res.seed_num = args_trace_info.tc_num  # 后续用于fuzz
            self.sbs_io_info[sb.subtree_id] = encode_res

    def single_encoding(self, subtree: "ASTNode") -> EncodeRes:  # type: ignore
        # 多个测试用例执行，不会有None的情况
        return self.sbs_io_info[subtree.subtree_id]

    def batch_encoding(self) -> List[Tuple]:
        return self.sbs_io_info.values()
