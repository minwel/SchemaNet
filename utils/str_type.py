import os
import sys
from pathlib import Path
import re
from typing import List, Tuple
from static.template import head, tail

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


class String:
    def __init__(self, text: str) -> None:
        self.text = text

    def __str__(self, ) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.text

    def get_text(self, ) -> str:
        return self.text


class PathString(String):
    def __init__(self, path: str):
        self.valid_f_names = [
            r"^[\u4e00-\u9fa5]+-\d+-\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}$",  # 早期我的命名
            r"^\d{8}_\d+_\d+_\d+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$",  # 孟霜给的数据的命名
        ]
        super().__init__(os.path.normpath(path))  # 自动规范化路径

    def __fspath__(self):
        return self.text

    def get_parent(self):
        """ 获取父文件夹 """
        return os.path.dirname(self.text)

    def get_filename(self):
        """ 如果是文件，可用于获取文件名，含扩展名 """
        return os.path.basename(self)

    def get_filename_without_ext(self):
        """ 如果是文件，可用于获取文件名，不含扩展名 """
        return os.path.splitext(self.get_filename())[0]

    def join(self, *other_paths):
        """ 用于与其他path连接 """
        return PathString(os.path.join(self.text, *other_paths))

    def get_prob_id(self):
        """ 如果是代码文件，可以用来获得代码的prob_id """
        # 杜涛-2900-2023-04-13_16_50_37
        # 20235102_1469_2749_1_2023-12-21_23-49-25
        if (is_valid := self.is_valid_codePath())[0]:
            if is_valid[1] == 0:
                return re.search(r"-(\d+)-", self.get_filename_without_ext()).group(1)
            elif is_valid[1] == 1:
                return re.search(r"^[^_]*_[^_]*_([^_]+)", self.get_filename_without_ext()).group(1)
            # else: # 后续可能更多
            # return self.get_filename_without_ext().split('-')[1]
        else:
            raise ValueError("文件名格式不正确")

    def is_valid_codePath(self):
        for idx, vfn in enumerate(self.valid_f_names):
            if re.match(vfn, self.get_filename_without_ext()):
                return True, idx
        # if re.match(r"[^-]*?-\d+", self.get_filename_without_ext()):
        #     return True
        return False, 0

    def is_valid(self, ):
        """ 判断路径是否存在 """
        return os.path.exists(self.text)

    def read(self, ):
        """ 读取文件内容 """
        if os.path.isfile(self):
            with open(self.text, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise FileNotFoundError

    def __eq__(self, value):
        if isinstance(value, str):
            return self.get_text() == value
        elif isinstance(value, PathString):
            return self.get_text() == value.get_text()


class CodeString(String):
    def __init__(self, code: str) -> None:
        super().__init__(code)

    def get_lines(self, ) -> List[str]:
        """ 返回code的每一行内容 """
        # 由于有的代码存在开头空行的情况，因此不应该strip()
        return self.text.split('\n')

    def get_code_line_count(self, ) -> int:
        """ 返回code的行数 """
        return len(self.get_lines())

    @staticmethod
    def join(*code_strings: Tuple["CodeString", str]):
        return CodeString('\n'.join([str(code) for code in code_strings]))

    def add_indented(self, n_indent: int) -> None:
        """ 增加n个缩进, 默认4个空格为一个缩进 """
        self.text = '\n'.join([' ' * 4 * n_indent + line for line in self.get_lines()])
        return self.text

    def get_exec_func_wrapped(self, inputs: List):
        """ 用于将该代码包装成函数，源代码，非子树代码，函数添加的是`def func(input):` """
        from copy import deepcopy

        mock = "{'input': lambda : next(input_iter)}"
        exce_str = f"input_iter = iter({inputs})\nfunc(**{mock})"

        return CodeString.join(
            CodeString("def func(input):"),
            deepcopy(self).add_indented(1),
            exce_str
        )

    def get_trace_wrapped(self, code_len_ori: int):
        """ 用于将该代码包装成可以执行变量跟踪的代码，跟踪结果保存到`res_json_path` """
        head_ = head % code_len_ori
        return CodeString.join(head_, self.get_text(), tail)
