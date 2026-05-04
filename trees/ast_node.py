import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from static.structs import DisRes, EncodeRes, EncodeResNew
import ast
import math
import astor
import numpy as np
from utils.tree import GraphRenderer
from utils.mathematics import get_cosine_dis
from utils.str_type import CodeString
from typing import Dict, List
from trees.visitor import NameVisitor, ConstantExtractor
from refactory.basic_framework.repair import MyMapper
from encoder.cent import CentEncoder
from encoder.io_encoder import IOEncoder
import copy

var_mapper = MyMapper()  # for trace
gr = GraphRenderer()  # for rendering



class ASTNode:
    """
    表示代码子树节点。
    """
    def __init__(
        self,
        ast_node: ast.AST,
        subtree_id: int = 0,  # start from 0
        code_id: int = 0,
        f_name: str = "",
        is_correct: bool = False,
    ):
        self.ast_node = ast_node
        self.subtree_id = subtree_id
        self.code_id = code_id
        self.f_name = f_name
        self.is_correct = is_correct
        self.type_1 = -1
        self.type_2 = -1
        self.dis_res_1 = DisRes()
        self.dis_res_2 = DisRes()
        self.tfidf = -1
        self.text = self.get_text()
        self.cal_tfidf_process = ""
        self.trace_map: EncodeRes = None
        self.trace_map_new: EncodeResNew = None
        # (入参[]，局部变量[])，这里初始化用于encode时过滤仅在函数内出现的所有变量
        # ，但在encode时还要更新vars，排除实际上不需要显式输入值的变量，如下标变量
        self.vars = self.get_vars()  
        # func_body在encode时再初始化
        self.func_body = None
        self.matched = False
        self.start_line = self.get_lineinfo()[0]
        self.end_line = self.get_lineinfo()[1]

    def get_io_info_new(
            self,
            executor: "Executor",  # type: ignore
            test_fuzzer: "TestFuzzer",  # type: ignore
            tc_num_fuzzed: int,
            in_vec: Dict = None,
        ):
        """
        为从`encoder`处获取的种子fuzz更多测试用例并执行，
        获取每个输入变量io向量， 

        :in_vec: None：self节点，非None：value节点，{}：也是value节点，但self节点
        没有输入变量
        """

        if in_vec == None and self.trace_map_new:  # self node
            return self.trace_map_new
        ## 1、为每个变量用多峰分布模糊生成更多测试用例
        constants = self.get_constants()  # 块中出现的常量也作为seed
        if in_vec == None:  # self node
            tc_num = self.trace_map.seed_num
            for _, test_seed in self.trace_map.params.items():
                # 正常为每个输入变量fuzz
                tc_num = test_fuzzer.fuzz(
                    self, test_seed, constants, tc_num_fuzzed
                )
        else:  # value node
            tc_num = len(next(iter(in_vec.values()))) if in_vec != {} else 0
            for param, test_seed in self.trace_map.params.items():
                if not param in in_vec:  # 只fuzz独有的变量
                    tc_num = test_fuzzer.fuzz(
                        self, test_seed, constants, tc_num_fuzzed
                    )

        ## 2、用新生成的所有测试用例执行当前子树代码片段，获取输出值
        in_info, out_info = DisUtil.get_out_info(
            self, tc_num, executor, in_vec
        )

        io_info_new = EncodeResNew(in_info, out_info, tc_num)

        if in_vec == None:  # self node
            self.trace_map_new = io_info_new

        return io_info_new

    def get_io_dis(
            self, value: "ASTNode", map_: dict, 
            executor: "Executor", test_fuzzer: "TestFuzzer",  # type: ignore
            tc_num_fuzzed: int,
        ):
        """ 计算两个子树的IO距离
        fuzz多个测试用例进行测试，再计算距离；
        """
        dis_res = DisRes(
            dis=DisUtil.MAX_DIS,
            _self=self.subtree_id,
            _value=value.subtree_id,
            _map=map_,
            vars={
                'self': self.vars,
                'value': value.vars
            },
        )

        ## 1、生成测试用例并执行获取输出（默认按self的生成）
        x_io_info: EncodeResNew = self.get_io_info_new(
            executor, test_fuzzer, tc_num_fuzzed
        )

        mapped = {map_.get(var, var): vals for var, vals in x_io_info.in_info.items()}
        y_io_info: EncodeResNew = value.get_io_info_new(
            executor, test_fuzzer, 
            tc_num_fuzzed,
            mapped
        )  # EncodeResNew.valid: 1：正常fuzz并编码；0：没有输入变量

        dis_res.encode_info = {
            "self": x_io_info,
            "value": y_io_info,
        }
        x_in, x_out = x_io_info.in_info, x_io_info.out_info
        y_in, y_out = y_io_info.in_info, y_io_info.out_info

        # 使用`HitoshiIO`的公式
        DisUtil.cal_HitoshiIO(
            dis_res=dis_res, map_=map_,
            x_in=x_in, y_in=y_in, x_out=x_out, y_out=y_out,
            tc_num_fuzzed=tc_num_fuzzed
        )

        return dis_res

    def get_dis(
            self,
            value: "ASTNode",
            encoder: "SubtreeEncoder", # type: ignore
            executor: "Executor" = None, # type: ignore
            test_fuzzer: "TestFuzzer" = None, # type: ignore
            tc_num_fuzzed: int = 15,  # 最终测试用例数量
            encoder_map: Dict[int, str] = None,
            **kwargs
        ) -> DisRes:
        """ 根据不同encoder计算两子树的距离

        :param value: 待比较的子树
        :param encoder: 子树编码器
        :param executor: 代码执行器
        :param test_fuzzer: 测试用例生成器
        """
        dis_res = DisRes()
        dis_res._self = self.subtree_id # 自己的id
        dis_res._value = value.subtree_id # 比较的子树的id

        if isinstance(encoder, CentEncoder):
            dis_res.dis = get_cosine_dis(
                encoder.single_encoding(self), encoder.single_encoding(value)
            )
        elif isinstance(encoder, IOEncoder):
            # 获取self节点和待比较的value节点的trace信息
            if not self.trace_map:
                self.trace_map = encoder.single_encoding(self)
            if not value.trace_map:
                value.trace_map = encoder.single_encoding(value)

            # 两个代码块任一无法编码（编译错误/超时），直接返回
            if not (self.trace_map.valid and value.trace_map.valid):
                return DisRes(note="compile error or timeout")

            # 获取变量映射表
            _map = var_mapper.get_vn_map(self, value, "func")

            dis_res = self.get_io_dis(
                value, _map, executor, test_fuzzer, tc_num_fuzzed
            )


        # 记录最终分类依据
        if encoder_map:
            encoder_map_reversed = {v: k for k, v in encoder_map.items()}
            tmp = {
                "CentEncoder": "cent",
                "IOEncoder": "io",
            }.get(str(encoder))
            dis_res_attr = f"dis_res_{encoder_map_reversed[tmp]}"

            if dis_res.dis < getattr(self, dis_res_attr).dis:  # 记录self的最优距离
                setattr(self, dis_res_attr, dis_res)
            if dis_res.dis < getattr(value, dis_res_attr).dis:  # 记录value的最优距离
                setattr(value, dis_res_attr, dis_res)

        return dis_res

    def get_text(self, ) -> str:
        """ 获取该子树的文本表示（formatted） """
        return ast.unparse(self.ast_node)

    def get_code_line_count(self, ) -> int:
        """ 获取node的代码行数 """
        return CodeString(self.text).get_code_line_count()

    def get_vars(self, ) -> list:
        in_vars, local_vars = map(list, NameVisitor.visit_and_filter(self.ast_node))
        return [in_vars, local_vars]

    def get_func_body(self, args: List[str] = None) -> CodeString:
        """
        生成函数签名，将子树包装为函数体, 返回函数体的文本表示(formatted)
        """
        if self.func_body:
            return self.func_body

        if args is None:
            # raise ValueError("args should not be None")
            args = []

        # 构造函数体
        func = ast.FunctionDef(
            name="func",
            # 加入**kargs，以方便传参的统一
            args=ast.arguments(args=args + ["**kwargs"], defaults=[]),
            # 这里又转换一遍是为了获取body属性
            # 只有parse完是Module类型才有body属性，但应该可以优化
            body=ast.parse(self.get_text()).body,
            decorator_list=[],
            returns=None,
        )
        return CodeString(astor.to_source(func))

    def get_constants(self, ):
        return ConstantExtractor.visit_and_extract(self.ast_node)

    def get_func_executable(self, args: Dict) -> CodeString:
        """ 获取可执行函数，包含：函数体 + 调用语句 + 输入变量值
        :args: 用于构造调用语句的输入参数;
        """
        return CodeString.join(self.get_func_body(), "func(**{})".format(args))

    def get_all_node_type(self, ):
        """ 用于辅助cent编码器记录节点类型 """
        ans = set()
        for node in ast.walk(self.ast_node):
            ans.add(gr.get_dynamic_value(node))
        return ans

    def get_node_n_edge(
        self,
        node: ast.AST,
        src: list,
        tgt: list,
    ):
        """ 用于辅助cent编码器对node的子节点连边 """
        for child in ast.iter_child_nodes(node):
            src.append(gr.get_dynamic_value(node))
            tgt.append(gr.get_dynamic_value(child))
            self.get_node_n_edge(child, src, tgt)

    def __eq__(self, value: object, cent_encoder, threshold: int = 3) -> bool:
        if isinstance(value, self):
            dis = np.linalg.norm(
                np.array(cent_encoder.get_single_tree_encoding(self))
                - np.array(cent_encoder.get_single_tree_encoding(value))
            )
            if dis < threshold:
                return True
            else:
                return False
        return False

    def __hash__(self) -> int:
        return hash(self.subtree_id)

    def __str__(self) -> str:
        if self.text:
            return self.text
        return self.get_text()

    def __repr__(self) -> str:
        if self.text:
            return self.text
        return self.get_text()

    def get_sub_trees(self, code2tree: Dict = {}) -> List["ASTNode"]:
        plans = []
        index = 0

        def add_block(nodes: List[ast.stmt]):
            nonlocal index
            if not nodes:
                return
            plans.append(
                ASTNode(
                    ast_node=ast.Module(body=nodes, type_ignores=[]),
                    subtree_id=self.subtree_id + index,
                    code_id=self.code_id,
                    f_name=self.f_name,
                    is_correct=self.is_correct,
                )
            )
            code2tree.setdefault(self.code_id, []).append(self.subtree_id + index)
            index += 1

        def contains_input_call(node: ast.AST) -> bool:
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "input":
                    return True
            return False
        
        def is_input_stmt(node):
            return (
                isinstance(node, (ast.Assign, ast.Expr)) and contains_input_call(node)
            )
    
        def handle_control_flow(node):
            """将控制流节点拆分为多个逻辑块"""
            if isinstance(node, ast.If):
                if_block = copy.deepcopy(node)
                if_block.orelse = [] 
                add_block([if_block])
                if node.orelse:
                    add_block(node.orelse)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                for_block = copy.deepcopy(node)
                for_block.orelse = [] 
                add_block([for_block])
                if node.orelse:
                    add_block(node.orelse)
            elif isinstance(node, ast.While):
                while_block = copy.deepcopy(node)
                while_block.orelse = [] 
                add_block([while_block])
                if node.orelse:
                    add_block(node.orelse)
            elif isinstance(node, ast.Try):
                if node.body:
                    try_block = copy.deepcopy(node)
                    try_block.handlers = [] 
                    try_block.orelse = [] 
                    try_block.finalbody = [] 
                    add_block([try_block])
                if node.handlers or node.orelse:
                    add_block(node.handlers + node.orelse)
                if node.finalbody:
                    add_block(node.finalbody)

        def recursive_traverse(nodes: List[ast.stmt]):
            i = 0
            while i < len(nodes):
                node = nodes[i]

                # === 1. 连续 input 块 ===
                if is_input_stmt(node):
                    block = [node]
                    i += 1
                    while i < len(nodes) and is_input_stmt(nodes[i]):
                        block.append(nodes[i])
                        i += 1
                    add_block(block)
                    continue

                # === 2. 连续赋值块（不含 input）===
                elif isinstance(node, (ast.Assign, ast.AugAssign)) and not is_input_stmt(node):
                    block = [node]
                    i += 1
                    while i < len(nodes) and isinstance(nodes[i], (ast.Assign, ast.AugAssign)) and not is_input_stmt(nodes[i]):
                        block.append(nodes[i])
                        i += 1
                    add_block(block)
                    continue

                # === 3. 控制流结构块 ===
                elif isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try)):
                    handle_control_flow(node)
                    # 对 control flow 的 body 和 orelse 递归处理
                    if hasattr(node, "body"):
                        recursive_traverse(node.body)
                    if hasattr(node, "orelse"):
                        recursive_traverse(node.orelse)
                    if isinstance(node, (ast.Try)):
                        for handler in node.handlers:
                            recursive_traverse(handler.body)
                        if node.finalbody:
                            recursive_traverse(node.finalbody)
                    i += 1
                    continue

                else:
                    # 其它语句单独成块
                    add_block([node])
                    i += 1

        # 开始处理当前 ast 节点的 body
        if isinstance(self.ast_node, ast.Module):
            recursive_traverse(self.ast_node.body)
        elif isinstance(self.ast_node, (ast.FunctionDef, ast.ClassDef)):
            recursive_traverse(self.ast_node.body)
        else:
            # fallback: treat this as a single-node block
            add_block([self.ast_node])

        return plans
    
    def get_lineinfo(self, ):
        start_line = float('inf')
        end_line = -1

        for node in ast.walk(self.ast_node):
            if hasattr(node, 'lineno'):
                start_line = min(start_line, node.lineno)
                end_line = max(end_line, node.lineno)

        if start_line == float('inf') or end_line == -1:
            print("No valid line info found.")
            return None
        
        return start_line, end_line

class DisUtil:
    MAX_DIS = 100  # 最大距离

    @staticmethod
    def get_sim(l1: List, l2: List) -> float:
        """ 计算两个列表的相似度 """
        sim_vec = [1 if x == y else 0 for x, y in zip(l1, l2)]
        return sum(sim_vec) / len(sim_vec)

    @staticmethod
    def get_out_info(
        node: ASTNode,
        tc_num: int,
        executor: "Executor",  # type: ignore
        in_vec: Dict = None,
    ):
        """ 
        从输入信息执行获取输出信息

        Param
        ---
        :node: 所属节点        
        :tc_num: 测试用例数量
        :executor: 执行器
        :in_vec: self节点的输入变量
        """

        in_info = {}
        out_info = {}

        if 'input' in node.text:
            return in_info, out_info

        # 代码基本信息
        sb_line_cnt = node.get_code_line_count()
        out_vars = node.trace_map.out_vars
        out_consts = node.trace_map.out_consts
        
        # 对每个测试用例执行
        for tc_idx in range(tc_num):
            ## 1、获取该测试用例的输入
            in_args = {}
            if in_vec == None:  # self node
                for arg, test_seed in node.trace_map.params.items():
                    in_args[arg] = test_seed.vals[tc_idx]
            else:  # value node
                for arg, test_seed in node.trace_map.params.items():
                    if arg in in_vec:  # 使用self的输入值
                        in_args[arg] = in_vec[arg][tc_idx]
                    else:  # 独有输入变量
                        in_args[arg] = test_seed.vals[tc_idx]
            # 加入最终输入in_info
            for arg in in_args:
                in_info.setdefault(arg, []).append(in_args[arg])

            ## 2、以该tc执行代码
            # 包装成可执行函数
            exec_code = node.get_func_executable(in_args)
            # 执行代码，获取变量跟踪结果
            tmp_res = executor.execute(
                code_str=exec_code,
                code_len_ori=sb_line_cnt,
                enable_trace=True
            )
            _idx_valid = 1
            if not tmp_res.is_executable:
                _idx_valid = -1
            else:
                # 该sb（node）执行该tc的变量跟踪信息
                tmp_trace_res = tmp_res.trace_res

            ## 3、获取该tc执行的输出值
            if _idx_valid == 1:
                end_line = max(
                    tmp_trace_res.keys(),  # lines
                    key=lambda x: tmp_trace_res[x]["index"],
                )
            # 添加变量
            for var in out_vars:
                if _idx_valid == 1:  # 编译成功
                    arg_out = tt[var] \
                    if var in (tt := tmp_trace_res[end_line]["end"]) \
                    else None  # 该tc未覆盖该var则添None
                else:  # 编译失败，全部-1
                    arg_out = _idx_valid
                # 添加到out_info，并补齐之前tc未覆盖的
                out_info.setdefault(var, [None] * tc_idx).append(arg_out)
            # 添加常量
            if out_consts:
                out_info.setdefault('const', []).append(out_consts)

        # 补齐之后tc未覆盖的
        for arg in out_info:
            if len(out_info[arg]) != tc_num:
                out_info[arg] += [None] * (tc_num - len(out_info[arg]))
        
        # 验证out_info的长度是否正确
        flag_sum = sum((len(v) for _, v in out_info.items()))
        if tc_num:
            assert flag_sum % tc_num == 0, "out_info 未对齐"
        
        return in_info, out_info

    @staticmethod
    def cal_HitoshiIO(
        dis_res: DisRes, map_: Dict,
        x_in: Dict, y_in: Dict, x_out: Dict, y_out: Dict,
        tc_num_fuzzed: int
    ):
        """
        目前是已经支持了输入变量不同的比较
        """
        beta = 3  
        default_noinputs_sim_i = 1
        default_noinputs_sim_o = 1

        sims = 0

        lens = []
        for info in (x_in, y_in, x_out, y_out):
            for _, vals in info.items():
                lens.append(len(vals))
        tc_num_effective = min(lens) if lens else 0
        tc_num_effective = min(tc_num_effective, tc_num_fuzzed)
        if tc_num_effective == 0:
            dis_res.dis = DisUtil.MAX_DIS
            return

        for invok in range(tc_num_effective):
            ## 1、计算sim_i
            x_args_mapped = set(
                (map_.get(x, x), str(y[invok])) for x, y in x_in.items()
            )
            y_args = set((x, str(y[invok])) for x, y in y_in.items())
            intersect_args = x_args_mapped & y_args
            union_args = x_args_mapped | y_args
            if union_args:
                sim_i = len(intersect_args) / len(union_args)
            else:  # 都没有输入变量
                sim_i = default_noinputs_sim_i

            ## 2、计算sim_o
            x_args_mapped = set(
                (map_.get(x, x), str(y[invok])) for x, y in x_out.items()
            )
            y_args = set((x, str(y[invok])) for x, y in y_out.items())
            intersect_args = x_args_mapped & y_args
            union_args = x_args_mapped | y_args
            if union_args:
                sim_o = len(intersect_args) / len(union_args)
            else:  # 都没有输出变量
                sim_o = default_noinputs_sim_o

            ## 3、公式
            sim = (
                (1 - beta * math.exp(sim_i)) * (1 - beta * math.exp(sim_o))
            ) / ((1 - beta * math.e) ** 2)
            sims += sim

        dis_res.dis = tc_num_effective / sims if sims else DisUtil.MAX_DIS
