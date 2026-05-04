import io
import keyword
import sys
import ast
import tokenize
import traceback
from typing import Set, Tuple
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import builtins

class NameVisitor(ast.NodeVisitor):
    """ 基于ast库的Name节点遍历器，主要用于过滤出输入变量 """
    # 内置函数和变量名
    builtin_names = dir(builtins)

    # 用于过滤非变量Name节点的辅助集合
    classes = set()
    functions = set()
    builtins_found = set()

    # 用于过滤非参数变量Name节点的辅助集合
    load_vars = set()
    store_vars = set()

    # 需要进一步判断
    to_justify = set()

    @classmethod
    def visit_and_filter(cls, node: ast.AST) -> Tuple[Set, Set]:
        """
        遍历所有节点，过滤非变量节点，返回变量集合，并进一步过滤出入参变量 

        Param
        ---
        :node: ast.AST
        
        Return
        ---
        :Tuple[Set, Set]: (入参变量集合，局部变量集合)
        """
        # 重置辅助集合
        cls.classes = set()
        cls.functions = set()
        cls.builtins_found = set()
        cls.load_vars = set()
        cls.store_vars = set()
        # cls.to_justify = set()

        # 遍历所有节点
        cls.visit(cls(), node)

        # # 处理to_justify中的节点
        # for x in cls.to_justify:
        #     if x not in cls.store_vars and x not in cls.load_vars:
        #         cls.load_vars.add(x)
        return cls.load_vars, cls.store_vars

    @classmethod
    def visit_ClassDef(cls, node):
        # 类定义中的名字是类名
        if node.name not in cls.builtin_names:
            cls.classes.add(node.name)
        cls.generic_visit(cls(), node)

    @classmethod
    def visit_FunctionDef(cls, node):
        # 函数定义中的名字是函数名
        if node.name not in cls.builtin_names:
            cls.functions.add(node.name)
        cls.generic_visit(cls(), node)

    @classmethod
    def visit_Call(cls, node):
        # 函数调用中的名字
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in cls.builtin_names:
                cls.builtins_found.add(func_name)
            else:
                cls.functions.add(func_name)
        cls.generic_visit(cls(), node)

    @classmethod
    def visit_Assign(cls, node):
        # 变量出现在赋值语句左侧
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id not in cls.builtin_names:
                if isinstance(target.ctx, ast.Load):
                    # if not (target.id in (cls.load_vars | cls.store_vars)):
                    cls.load_vars.add(target.id)  # 被读取的变量
                elif isinstance(target.ctx, ast.Store):
                    # if not (target.id in (cls.load_vars | cls.store_vars)):
                    cls.store_vars.add(target.id)  # 被赋值的变量
        cls.generic_visit(cls(), node)

    @classmethod
    def visit_AugAssign(cls, node):
        # 检查并打印增强赋值的目标变量
        if isinstance(node.target, ast.Name):  # 确保是变量名
            # print(f"AugAssign target: {node.target.id}")  # 输出变量名
            # if not (node.target.id in (cls.load_vars | cls.store_vars)):
            cls.load_vars.add(node.target.id)
            cls.store_vars.add(node.target.id)
        # 继续访问其他节点
        cls.generic_visit(cls(), node)

    @classmethod
    def visit_Name(cls, node):
        # Name 节点出现在表达式中时，如果它不是类名或函数名，认为它是变量名
        if node.id in cls.builtin_names:
            cls.builtins_found.add(node.id)
        elif node.id not in cls.functions and node.id not in cls.classes:
            if isinstance(node.ctx, ast.Load):
                cls.load_vars.add(node.id)
            elif isinstance(node.ctx, ast.Store):
                cls.store_vars.add(node.id)
        cls.generic_visit(cls(), node)


class VariableFinder(ast.NodeVisitor):
    def __init__(self, target_variable):
        self.target_variable = target_variable
        self.found = False

    @classmethod
    def find(cls, node, target_variable):
        # 创建实例并开始查找
        instance = cls(target_variable)
        instance.visit(node)
        return instance.found

    def visit_Name(self, node):
        # 如果变量名匹配目标变量，标记为已找到
        if node.id == self.target_variable:
            self.found = True
        # 继续遍历子节点
        self.generic_visit(node)

def extract_variables_from_unparsable_code(code):
    """ 用tokenize库分析后文出现的变量 """
    
    try:
        # 确保代码是字符串类型
        if not isinstance(code, str):
            raise ValueError("Input code must be a string.")
        
        # 将代码字符串转换为字节流
        code_stream = io.StringIO(code).readline
        variables = set()

        # 使用 tokenize 逐词扫描
        for token in tokenize.generate_tokens(code_stream):
            token_type, token_string = token.type, token.string
            
            # 如果是标识符并且不是关键字，可能是变量
            if token_type == tokenize.NAME and not keyword.iskeyword(token_string):
                variables.add(token_string)
        
        return variables
    except Exception as e:
        # print(f"Error processing code: {e}")
        traceback.print_exc()
        return set()


class ReturnValueAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.vars = set()
        self.consts = set()

    def visit_Call(self, node):
        # 处理 print 函数调用
        if isinstance(node.func, ast.Name) and node.func.id == 'print':
            for arg in node.args:
                self.extract_variable_or_constant(arg)
        self.generic_visit(node)
    
    def visit_Return(self, node):
        # 处理 return 语句
        if node.value:
            self.extract_variable_or_constant(node.value)
        self.generic_visit(node)

    def extract_variable_or_constant(self, expr):
        # 可以根据需要继续扩展处理其他类型的节点，如函数调用等
        # 提取表达式中的变量或常量
        if isinstance(expr, ast.Name):  # 变量
            self.vars.add(expr.id)
        elif isinstance(expr, ast.Constant):  # 常量
            self.consts.add(expr.value)
        elif isinstance(expr, ast.BinOp):  # 处理二元操作符
            self.extract_variable_or_constant(expr.left)
            self.extract_variable_or_constant(expr.right)
    
    @classmethod
    def analyze_code(cls, code):
        # 将代码解析为 AST，并遍历它
        tree = ast.parse(code)
        analyzer = cls()
        analyzer.visit(tree)

        return {
            "vars": analyzer.vars,
            "consts": analyzer.consts
        }


class NameTransformer(ast.NodeTransformer):
    """ 从ast树的层面在树中转换节点变量名 """
    def __init__(cls, var_mapper):
        super().__init__()
        cls.var_mapper = var_mapper

    def visit_Name(cls, node: ast.Name):
        if node.id in cls.var_mapper:
            node.id = cls.var_mapper[node.id]
        # return node
        return cls.generic_visit(cls(), node)

    def visit_FunctionDef(cls, node: ast.FunctionDef):
        for arg in node.args["args"]:
            if arg.arg in cls.var_mapper:
                arg.arg = cls.var_mapper[arg.arg]
        # return node
        return cls.generic_visit(cls(), node)


class ConstantExtractor(ast.NodeVisitor):
    constants = {}

    @classmethod
    def visit_and_extract(cls, node: ast.AST) -> Set:
        cls.constants = {}
        cls.visit(cls(), node)
        return cls.constants

    @classmethod
    def visit_Constant(cls, node: ast.Constant):
        # Python 3.8+

        cls.constants.setdefault(type(node.value).__name__, set()).add(node.value)

    @classmethod
    def visit_Num(cls, node: ast.Num):
        # Python < 3.8
        cls.constants.setdefault(type(node.n).__name__, set()).add(node.n)

    @classmethod
    def visit_Str(cls, node: ast.Str):
        # Python < 3.8
        cls.constants.setdefault(type(node.s).__name__, set()).add(node.s)

    @classmethod
    def visit_Tuple(cls, node: ast.Tuple):
        # 捕获元组中的常量
        for elt in node.elts:
            if isinstance(elt, ast.Constant):
                cls.constants.setdefault(type(elt.value).__name__, set()).add(elt.value)

