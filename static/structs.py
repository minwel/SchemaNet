from typing import Dict, List, Union
from dataclasses import dataclass, field
from typing import List, Tuple, Set


@dataclass
class ExecRes:
    """ 
    执行结果封装类
    检验时，需先检验is_executable，再检验is_correct，不可反过来

    `is_executable` : 代码是否可执行（无编译错误和超时问题）
    `is_correct` : 代码是否正确（通过所有测试用例）
    `code_output` : 代码在可执行情况下每个测试用例的输出结果
    `tc_output` : 每个测试用例的标准输出
    `trace_res` : 跟踪结果
    """

    is_tc_included: int = 1
    # 编译正确性
    is_executable: int = 1
    exception: BaseException = None
    # 逻辑正确性
    is_correct: int = 1
    code_output: List[Union[int, str]] = field(default_factory=list)
    tc_output: List[Union[int, str]] = field(default_factory=list)
    pass_info: List[int] = field(default_factory=list)

    # 覆盖信息
    cov_info: List[Set[int]] = field(default_factory=list)
    
    # 变量值跟踪序
    trace_res: Union[List[Dict], Dict] = field(default_factory=list)
    trace_lines_seq: Union[List[List[int]], List] = field(default_factory=list)

    def __str__(self, ):
        return self.__dict__.__str__()

@dataclass
class LineTraceInfo:
    """

    param:
    ---
    :start: 该行最先出现的位置的变量和值，dict表示，如果该行没有被覆盖，则为None
    :end: 该行最后出现的位置的变量和值，dict表示，如果该行没有被覆盖，则为None
    :index: 该行最后出现的绝对位置序号
    """
    start: Dict = field(default_factory=dict)
    end: Dict = field(default_factory=dict)
    index: int = 0

    @classmethod
    def from_dict(cls, args: dict):
        return cls(**args)

@dataclass
class TraceRes:
    lines_trace_info: Dict[int, List[LineTraceInfo]] = field(default_factory=dict)
    lines_seq: List[List[int]] = field(default_factory=list)
    tc_num: int = 0

    def get_line_inputargs(self, lineno: int):
        if not lineno in self.lines_trace_info:
            raise IndexError(f"sb line {lineno} not in trace_res")
        return self.lines_trace_info[lineno]


@dataclass
class TestSeed:
    param_name: str = ""
    _type: str = "int"
    fuzzed: bool = False
    vals: List = field(default_factory=list)

@dataclass
class EncodeRes:
    """ 原始运行时编码结果，由题目测试用例决定，没有经过fuzz扩充
    
    Params
    ---
    :valid: 1: 有效编码，0: bug code
    :in_info: 原始测试用例（默认5个）运行时输入变量和值，list封装5个测试用例，每个
    测试用例下的变量和值用dict封装
    :out_info: 对原始测试用例运行时分析后得到的输出变量及其值，目前没有用，主要是
    确定输出变量有哪些
    :out_vars: 对原始测试用例运行时分析得出的该块输出变量，主要用的是这个
    :out_consts: 对原始测试用例运行时分析后得到的输出常量
    :`params`: 原始测试用例中输入变量及其值转换为`TestSeed`表示，方便后面直接传给
    EncodeResNew
    """
    valid: int = 1  # bug code
    in_info: List[Dict] = field(default_factory=list)
    out_info: List[Dict] = field(default_factory=list)
    out_vars: Set = field(default_factory=set)
    out_consts: Set = field(default_factory=set)
    params: Dict[str, TestSeed] = field(default_factory=dict)
    seed_num: int = 0


@dataclass
class EncodeResNew:
    """ fuzz扩充测试用例之后的编码结果

    Params
    ---
    :in_info: 经过fuzz扩充后的输入变量和值，封装为TestSeed类
    :out_info: 经过fuzz扩充后的输出变量和值，封装为TestSeed类
    :tc_num: 测试用例数量
    """
    in_info: Dict[str, TestSeed] = field(default_factory=dict)
    out_info: Dict[str, TestSeed] = field(default_factory=dict)
    tc_num: int = 0


@dataclass
class DisRes:
    dis: float = float('inf')  # 初始化为正无穷
    _self: int = -1
    _value: int = -1
    _map: Dict = field(default_factory=dict)
    vars: Tuple = None
    encode_info: Dict = field(default_factory=dict)
    note: str = ""


