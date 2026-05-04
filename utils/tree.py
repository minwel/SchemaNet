import ast
import json
import os
import re
import sys
import zss
from pathlib import Path
import xmind

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
import numbers
import uuid
import graphviz as gv
import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

class TreeParser:
    def __init__(self) -> None:
        # Setup tree-sitter
        PY_LANGUAGE = Language(tspython.language())
        self.py_parser = Parser(PY_LANGUAGE)
    
    def parse(self,code: str) -> Node:
        return self.py_parser.parse(bytes(code, 'utf8')).root_node


class GraphRenderer:
    """
    this class is capable of rendering data structures consisting of
    dicts and lists as a graph using graphviz
    """

    graphattrs = {
        "labelloc": "t",
        "fontcolor": "white",
        "bgcolor": "#333333",
        "margin": "0",
    }

    nodeattrs = {
        "color": "white",
        "fontcolor": "white",
        "style": "filled",
        "fillcolor": "#006699",
    }

    edgeattrs = {
        "color": "white",
        "fontcolor": "white",
    }

    _graph = None
    _rendered_nodes = None

    @staticmethod
    def _escape_dot_label(str):
        return (
            str.replace("\\", "\\\\")
            .replace("|", "\\|")
            .replace("<", "\\<")
            .replace(">", "\\>")
        )

    @staticmethod
    def get_dynamic_value(node: ast.AST):
        """ 动态获取ast节点的实际值，区分常量值和其他变量或函数节点的值 """
        if isinstance(node, ast.Name):
            # 变量不区分
            return node.__class__.__name__
        if isinstance(node, ast.Attribute):
            # 函数区分
            return node.attr
        for attr in ['value', 'id', 's', 'n', 'attr']:
            if hasattr(node, attr):
                node_attr = getattr(node, attr)
                if isinstance(node_attr, ast.AST):
                    # 其他类不区分
                    return node.__class__.__name__
                else:
                    # 常量区分
                    return str(node_attr)
        return node.__class__.__name__

    def _render_node(self, node):
        if isinstance(node, (str, numbers.Number)) or node is None:
            node_id = uuid()
        else:
            node_id = id(node)
        node_id = str(node_id)

        if node_id not in self._rendered_nodes:
            self._rendered_nodes.add(node_id)
            if isinstance(node, dict):
                self._render_dict(node, node_id)
            elif isinstance(node, list):
                self._render_list(node, node_id)
            elif isinstance(node, ast.AST):
                self._render_ast(node, node_id)
            elif isinstance(node, zss.Node):
                self._render_zss(node, node_id)
            else:
                self._graph.node(node_id, label=self._escape_dot_label(str(node)))

        return node_id

    def _render_dict(self, node, node_id):
        self._graph.node(node_id, label=node.get("node_type", "[dict]"))
        for key, value in node.items():
            if key == "node_type":
                continue
            child_node_id = self._render_node(value)
            self._graph.edge(node_id, child_node_id, label=self._escape_dot_label(key))

    def _render_list(self, node, node_id):
        self._graph.node(node_id, label="[list]")
        for idx, value in enumerate(node):
            child_node_id = self._render_node(value)
            self._graph.edge(
                node_id, child_node_id, label=self._escape_dot_label(str(idx))
            )
    
    def _render_ast(self, node: ast.AST, node_id):
        """ 当节点类型为ast节点 """
        self._graph.node(node_id, label=GraphRenderer.get_dynamic_value(node))
        for idx, value in enumerate(ast.iter_child_nodes(node)):
            child_node_id = self._render_node(value)
            self._graph.edge(node_id, child_node_id, label=self._escape_dot_label(str(idx)))
    
    def _render_zss(self, node: zss.Node, node_id):
        """ 当节点类型为zss节点 """
        self._graph.node(node_id, label=node.label)
        for idx, value in enumerate(node.children):
            child_node_id = self._render_node(value)
            self._graph.edge(node_id, child_node_id, label=self._escape_dot_label(str(idx)))

    def render(self, data, out_path: str, *, label=None):
        # 函数参数中的 * 是一个特殊的语法，表示之后的参数必须以 关键字参数 的形式传递（即明确指定参数名）。这被称为 强制关键字参数。
        # create the graph
        graphattrs = self.graphattrs.copy()
        if label is not None:
            graphattrs["label"] = self._escape_dot_label(label)
        graph = gv.Digraph(
            graph_attr=graphattrs, node_attr=self.nodeattrs, edge_attr=self.edgeattrs
        )

        # recursively draw all the nodes and edges
        self._graph = graph
        self._rendered_nodes = set()
        self._render_node(data)
        self._graph = None
        self._rendered_nodes = None

        # display the graph
        out_dir = os.path.dirname(out_path)
        f_name = os.path.basename(out_path).split('.')
        graph.render(
            directory=out_dir,
            filename=f_name[0],
            format=f_name[1],
            view=False,
        )




