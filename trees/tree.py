import sys
import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from utils.executor import Executor
from utils.other import ConfigManager
from typing import List, Tuple, Union, Dict
from trees.ast_node import ASTNode


class Trees:
    def __init__(
        self,
        executor: Executor = None,
        config_manager: ConfigManager = None,
    ) -> None:
        """
        加载代码处理为子树

        :param parser: 
        :param trees: 所有子树
        :param code2tree: 代码到所包含的子树的映射
        :param code_src: 只能输入 `包含代码的文件夹路径` 或 `代码字符串的列表`
        :param
        """
        self.executor = executor
        self.config_manager = config_manager

        self.trees: List[ASTNode] = []  # 子树
        self.code2tree: Dict[int, List[int]] = {}  # code_id到sbs的映射
        self.tree_index = 0
        self.code_index = 0
        self.failed_code = []

    def add_batch(self, code_src: Union[str, List], is_correct: bool = True):
        for f_name, code_str in code_src:
            """ 加入单个代码的子树 """
            if not code_str.strip():
                return
            _, subtrees = self._get_sbs(code_str, f_name, is_correct)
            self.trees += subtrees
            self.tree_index += len(subtrees)
            self.code_index += 1

    def _get_sbs(
        self, code_str: str, f_name: str, is_correct: bool
    ) -> Tuple[ASTNode, List[ASTNode]]:
        """ 获取一个代码的ast并分割为子树 """
        code_tree = ASTNode(
            ast_node=ast.parse(code_str),
            subtree_id=self.tree_index,
            code_id=self.code_index,
            f_name=f_name,
            is_correct=is_correct,
        )
        subtrees = code_tree.get_sub_trees(
            self.code2tree
        )  # 将这个代码的ast树分割为子树
        return code_tree, subtrees


    def init_attributes(self, ):
        """ 重置 """
        self.__init__(
            executor=self.executor,
            config_manager=self.config_manager,
        )

    def __len__(self,):
        return len(self.trees)
