# 代码执行器、批量处理代码
import pickle
import sys
import threading
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from typing import Dict
import io
import trace
import signal
from utils.str_type import CodeString
from contextlib import contextmanager
from utils.exceptions import TimeoutException
from static.structs import ExecRes
TRACE_TMP_PKL = PROJECT_ROOT / "tmp" / "trace_tmp.pkl"
TRACE_TMP_PKL.parent.mkdir(parents=True, exist_ok=True)

def timeout_handler(signum, frame):
    raise TimeoutException
signal.signal(signal.SIGALRM, timeout_handler)

@contextmanager
def execute_decorator(res, my_stdout: io.StringIO, tracer, timeout):
    """ 代码执行装饰器
    
    负责异常处理，python的异常结构如下：
    ```
    BaseException
    ├── SystemExit
    ├── KeyboardInterrupt
    ├── GeneratorExit
    └── Exception
        ├── TypeError
        ├── ValueError
        ├── ...
    ```
    因此 `SystemExit` 异常需要单独捕获。另外还有代码超时判断。
    执行代码的上下文处理，包括输出流的重定向，计时器等。

    另外需要注意：
    如果在多线程环境中执行代码，`signal.alarm` 并不适合。如果这样，
    可以考虑使用 `threading.Timer` 或类似的方法来实现超时控制。

    """
    try:
        sys.stdout = my_stdout
        res.is_executable = 0
        signal.alarm(timeout)
        yield
    except SystemExit as se:  # 代码中含有exit()
        res.code_output = str(se)
        res.exception = se
    except TimeoutException as te:  # 代码执行超时
        res.code_output = str(te)
        res.exception = te
    except Exception as e:  # 其他常规异常
        res.code_output = str(e)
        res.exception = e
    else:
        res.code_output = my_stdout.getvalue()
        res.is_executable = 1
    finally:
        # 重置计时器和my_stdout
        try:
            signal.alarm(0)
            sys.stdout = sys.__stdout__
        except Exception:
            pass

def work(
        code_str, my_stdout, code_len_ori,
        offset: int = 1, timeout: int = 1
    ):
    """ 执行代码 """
    # 使用trace.Trace库运行代码并统计cov
    tmp_res = ExecRes()
    tracer = trace.Trace(count=True, trace=False)
    with execute_decorator(
        tmp_res, my_stdout, tracer, timeout
    ):
        cmd = f'exec("""{code_str.get_text()}""")'
        tracer.run(cmd)
    
    # 筛除原代码范围之外的行（因为加了def func行和调用行）
    tmp_res.cov_info = set(
        k[1] - offset  # 行数重置
        for k, v in tracer.counts.items()
        if offset < k[1] <= offset + code_len_ori
    )
    
    return tmp_res

class Executor:
    """ 代码执行器 """
    def __init__(
            self,
            test_cases: Dict,
            config_manager: "ConfigManager"  # type: ignore
        ):
        self.config = config_manager.config
        self.test_cases = test_cases  
        
    def execute(
        self,
        code_str: CodeString,
        offset: int = 1,
        code_len_ori: int = 0,
        enable_trace: bool = False,
    ) -> ExecRes:
        """ 
        执行一段代码，不包含模拟输入和变量跟踪

        Param
        ---
        :code_str: 可执行代码字符串，需要有def func头，以及func(**...)调用执行语句，
        也就是包装为可执行代码后的字符串；
        :offset: 偏移量，即原始代码首行前面加了多少行额外代码，如def func头，或者
        其他用于测试目的的代码；
        :code_len_ori: 最原始的代码行数（包装为可执行代码之前的原始代码）；
        :enable_trace：是否需要进一步包装成变量跟踪代码，包装后返回结果应
        只看trace_res字段；

        Return
        ---
        :ExecRes
        """
        # 如果需要变量跟踪，则包装成可追踪代码
        if enable_trace:
            code_str = code_str.get_trace_wrapped(code_len_ori)
        # 超时时间阈值
        timeout = self.config['timeout']
        try:
            # 执行代码
            res = work(code_str, io.StringIO(), code_len_ori, offset, timeout)
        except TimeoutException as te:
            # 代码执行超时
            res = ExecRes()
            res.is_executable = 0
            res.exception = te


        # 如果需要变量跟踪， 应该只关注trace_res，且只有一个
        if enable_trace and res.is_executable:
            with open(TRACE_TMP_PKL, 'rb') as f:
                try:
                    trace_f = pickle.load(f)
                except EOFError:
                    res.is_executable = 0
                    res.exception = Exception("文件为空或损坏")
                    return res
                res.trace_lines_seq = trace_f['lines_seq']
                res.trace_res = {
                    int(line): args_kv
                    for line, args_kv in trace_f.items()
                    if line != 'lines_seq'
                }
        return res

    def execute_code(
        self, code_str: CodeString, prob_id: str, enable_trace: bool = False
    ) -> ExecRes:
        """ 输入源代码和题目号，然后执行；
        
        - `code_str`: 代码字符串
        - `prob_id`: 题目号
        - `enable_trace`: 是否进行变量值跟踪
        - `return`: `ExecRes`
        """
        if not isinstance(code_str, CodeString):
            raise TypeError(f'param code_str expected class: CodeString, {type(code_str)} found.')

        # 记录原始代码信息
        code_str_ori = code_str
        code_len_ori = code_str.get_code_line_count()

        # 获取该题目的测试用例
        if not prob_id in self.test_cases:  # 该题目测试用例未导入
            return ExecRes(is_tc_included=0)
        tcs_input = self.test_cases[prob_id][0]  # ['[1,2,3,4,5,6,7,8,9]', '[2,4,6,8,10]', '[1,3,5,7,9,11,23,29]']
        tcs_output = self.test_cases[prob_id][1]  #  ['[2, 3, 5, 7]', '[2]', '[3, 5, 7, 11, 23, 29]']

        # 执行
        res = ExecRes()
        for tc_input, tc_output in zip(tcs_input, tcs_output):
            # 输入测试用例
            inputs = tc_input.split('\n')
            # 包装成可执行代码
            code_str = code_str_ori.get_exec_func_wrapped(inputs)
            # 执行
            tmp_res = self.execute(
                code_str=code_str, code_len_ori=code_len_ori
            )

            if tmp_res.is_executable:  # 可执行
                if tmp_res.code_output.strip() != tc_output:
                    res.is_correct = 0  # 一个测试用例不通过即代码不正确
                    res.pass_info.append(0)  # 记录每个测试用例的通过情况
                else:
                    res.pass_info.append(1)
                res.code_output.append(tmp_res.code_output.strip())
                res.tc_output.append(tc_output)
                res.cov_info.append(tmp_res.cov_info)          
            else:  
                # PARAM：含编译问题直接退出，不再跑剩下的用例
                res.is_executable = 0
                res.exception = tmp_res.exception  # 记录exception供外部调用
                break
            # 如果需要变量跟踪
            if enable_trace:
                # 再执行
                tmp_res = self.execute(
                    code_str=code_str,
                    code_len_ori=code_len_ori,
                    enable_trace=True,
                )
                # 只读取结果变量跟踪结果
                res.trace_res.append(
                    tmp_res.trace_res
                    if tmp_res.is_executable  # 变量跟踪有可能导致超时从而编译错误
                    else tmp_res.is_executable
                )
                res.trace_lines_seq.append(
                    tmp_res.trace_lines_seq
                    if tmp_res.is_executable
                    else tmp_res.is_executable
                ) 
        return res

if __name__ == "__main__":
    code_str = """
a=eval(input()) 
aver=sum(a)/len(a)
if sum(a)%len(a)==0:
    print(aver)
else:
    print('%.2f'%aver)
"""
    from utils.dataloader import TestCasesLoader
    from utils.other import ConfigManager
    from utils.file import read_yaml
    config = read_yaml("config.yml")
    test_cases = TestCasesLoader.get_test_cases("data/testcase")
    config_manager = ConfigManager(config)
    executor = Executor(test_cases, config_manager)
    code_ = CodeString(code_str)

    res = executor.execute_code(code_, "3004", True)
    print(res)
