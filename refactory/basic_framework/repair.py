import sys
import os
sys.path.append(os.path.dirname(__file__))
import random
from refactory.basic_framework.hole_injection import (
    get_cfs_map,
    get_token_list,
)

class BlockRepair:
    def __init__(self, ) -> None:
        pass

    def __is_equal(self, object_a, object_b):
        if str(type(object_a)) == str(type(object_b)):
            if object_a == object_b:
                return True
            else:
                return False
        else:
            close_type_list = ["<class 'list'>", "<class 'tuple'>"]
            if str(type(object_a)) in close_type_list and \
                    str(type(object_b)) in close_type_list:
                if list(object_a) == list(object_b):
                    return True
                else:
                    return False
            else:
                return False

    def __is_hist_equal(self, hist_a, hist_b):
        """ 用于比较两个trace_list是否完全相同 """
        if len(hist_a) != len(hist_b):
            return False
        for i in range(len(hist_a)):
            if not self.__is_equal(hist_a[i], hist_b[i]):
                return False
        return True

    def __def_use_analysis(self, bb_list, stat_vari_names):
        def_map = {}
        use_map = {}
        for bb_idx in range(len(bb_list)):
            bb = bb_list[bb_idx]
            line_list = bb.split("\n")[:-1]
            for line in line_list:
                token_list = get_token_list(line)

                assign_idx = -1
                for i in range(len(token_list)):
                    token = token_list[i]
                    if token.string in ["=", "+=", "-=", "*=", "/="]:
                        assign_idx = i
                        break

                for i in range(len(token_list)):
                    token = token_list[i]
                    if token.string in stat_vari_names:
                        if i < assign_idx:
                            if token.string not in def_map.keys():
                                def_map[token.string] = set()
                            def_map[token.string].add(bb_idx)
                        elif i > assign_idx:
                            if token.string not in use_map.keys():
                                use_map[token.string] = set()
                            use_map[token.string].add(bb_idx)
        return def_map, use_map

    def get_vn_map(self, self_node: "ASTNode", value_node: "ASTNode", func_name):  # type: ignore
        self_code = self_node.get_func_body().get_text()
        bug_bb_list, _, _ = get_cfs_map(self_code)[func_name]

        value_code = value_node.get_func_body().get_text()
        corr_bb_list, _, _ = get_cfs_map(value_code)[func_name]

        base_map = self.__get_vn_map_core(
            (self_node.trace_map.in_info, self_node.trace_map.out_info), 
            (value_node.trace_map.in_info, value_node.trace_map.out_info),
            bug_bb_list, corr_bb_list,
            self_node.vars, value_node.vars,
        )

        vn_map = {}
        for vn_a, cand_list in base_map.items():
            if isinstance(cand_list, list):
                vn_map[vn_a] = random.sample(cand_list, 1)[0]
            else:
                vn_map[vn_a] = cand_list

        return vn_map

    def __get_vn_map_core(self, trace_map_a, trace_map_b,
                        bb_list_a, bb_list_b,
                        vars_a, vars_b):
        base_map = {}
        vari_names_a = vars_a[0] + vars_a[1]  
        vari_names_b = vars_b[0] + vars_b[1]

        if len(vars_a[0]) == len(vars_b[0]):
            for i in range(len(vars_a[0])):
                base_map[vars_a[0][i]] = [vars_b[0][i]]

        for vn_a in vari_names_a: 
            for vn_b in vari_names_b:
                is_matched = True
                flag = False
                tc_num = min(len(trace_map_a[0]), len(trace_map_b[0]))
                for tc_id in range(tc_num):  
                    a_in = vn_a in trace_map_a[0][tc_id].keys() if trace_map_a[0][tc_id] else None
                    b_in = vn_b in trace_map_b[0][tc_id].keys() if trace_map_b[0][tc_id] else None
                    if not a_in and not b_in:  
                        flag = True
                        continue  
                    elif a_in and not b_in:  
                        is_matched = False
                        break
                    elif not a_in and b_in:
                        is_matched = False
                        break
                    else: 
                        a_trace_list = (
                            trace_map_a[0][tc_id][vn_a],
                            (
                                trace_map_a[1][tc_id][vn_a]
                                if trace_map_a[1][tc_id] and vn_a in trace_map_a[1][tc_id]
                                else None
                            ),
                        )  # (输入值，输出值)
                        b_trace_list = (
                            trace_map_b[0][tc_id][vn_b],
                            (
                                trace_map_b[1][tc_id][vn_b]
                                if trace_map_b[1][tc_id] and vn_b in trace_map_b[1][tc_id]
                                else None
                            ),
                        )
                        if not self.__is_hist_equal(a_trace_list, b_trace_list):
                            is_matched = False
                            break

                # 全匹配且不是全都not a_in and not b_in的情况才映射
                if is_matched and not flag:
                    # if vn_a not in base_map.keys():  # 考虑函数参数列表以外的变量
                    #     base_map[vn_a] = []
                    # base_map[vn_a].append(vn_b)  # 可能有多个matched的变量，因此用list
                    base_map[vn_a] = [vn_b]  # 还是改为直接覆盖修正比较好

        # 2、Map variables based on def-use analysis
        def_map_a, use_map_a = self.__def_use_analysis(bb_list_a, vari_names_a)
        def_map_b, use_map_b = self.__def_use_analysis(bb_list_b, vari_names_b)
        for vn_a in vari_names_a:
            for vn_b in vari_names_b:
                if vn_a not in base_map.keys() and \
                    vn_b not in self.get_mapped_vari(base_map):
                    if vn_a in def_map_a.keys() and vn_b in def_map_b.keys():
                        if def_map_a[vn_a] == def_map_b[vn_b]:
                            if vn_a not in base_map.keys():
                                base_map[vn_a] = []
                            base_map[vn_a].append(vn_b)
                    elif vn_a in use_map_a.keys() and vn_b in use_map_b.keys():
                        if use_map_a[vn_a] == use_map_b[vn_b]:
                            if vn_a not in base_map.keys():
                                base_map[vn_a] = []
                            base_map[vn_a].append(vn_b)

        # Using close name str to map variables
        for vn_a in vari_names_a:
            for vn_b in vari_names_b:
                if vn_a not in base_map.keys() and \
                    vn_b not in self.get_mapped_vari(base_map) and \
                    vn_a == vn_b:
                    if vn_a not in base_map.keys():
                        base_map[vn_a] = []
                    base_map[vn_a].append(vn_b)

        # Process residual variables
        for bvn in vari_names_a:
            if bvn not in base_map.keys():
                # value节点没有对应的变量，于是写作value_{}
                base_map[bvn] = ["value_" + bvn]

        for cvn in vari_names_b:
            if cvn not in self.get_mapped_vari(base_map):
            # if cvn not in list(base_map.values()):
                # self节点没有对应的变量，于是写作self_{}
                base_map["self_" + cvn] = [cvn]

        return base_map

    def get_mapped_vari(self, base_map):
        mapped_vari_list = []
        for cand_list in base_map.values():
            mapped_vari_list.extend(cand_list)
        return set(mapped_vari_list)


class MyMapper(BlockRepair):
    def __init__(self, ):
        pass
