# Developers:   Yang Hu, et al.
# Email:    huyang0905@gmail.com

import os
from basic_framework.exec import run_program_to

class Tester:
    def __init__(self, ques_dir_path):
        self.__ques_dir_path = ques_dir_path
        self.__ans_dir_path = ques_dir_path + "/ans"

        self.__input_dict = self.__ext_case_path(self.__ans_dir_path, "input")
        self.__output_dict = self.__ext_case_path(self.__ans_dir_path, "output")
        if len(list(self.__input_dict.keys())) == 0 or \
                len(list(self.__output_dict.keys())) == 0:
            raise Tester.NoTestCaseException()

        self.__front_code = ""
        # if os.path.isfile(ques_dir_path + "/code/global.py"):
        #     with open(ques_dir_path + "/code/global.py", "r") as f:
        #         self.__front_code += f.read()
        #         self.__front_code += "\n"

        self.__end_code = ""
        # if os.path.isfile(ques_dir_path + "/code/global_append.py"):
        #     with open(ques_dir_path + "/code/global_append.py", "r") as f:
        #         self.__end_code += f.read()
        #         self.__end_code += "\n"

    class NoTestCaseException(Exception):
        pass

    def __ext_case_path(self, dir_path, type_name):
        res = {}
        for file_name in os.listdir(dir_path):
            curr_path = dir_path + "/" + file_name
            if os.path.isdir(curr_path):
                part_res = self.__ext_case_path(curr_path, type_name)
                for k, v in part_res.items():
                    res[k] = v
            elif os.path.isfile(curr_path):
                mytype = file_name.split("_")[0]

                tc_id = file_name[len(mytype)+1:].split(".")[0]
                
                if mytype == type_name:
                    res[tc_id] = curr_path
        return res

    def get_tc_id_list(self):
        tc_id_list = list(self.__input_dict.keys())
        tc_id_list.sort()
        return tc_id_list

    def run_tc(self, code, tc_id, timeout):
        input_path = self.__input_dict[tc_id]
        entry_code = ""
        with open(input_path, "r") as f:
            entry_code += f.read()

        output_path = self.__output_dict[tc_id]
        exp_output = ""
        with open(output_path, "r") as f:
            try:
                exp_output += str(eval(f.read().strip())) + "\n"
            except:
                exp_output += f.read().strip() + "\n"

        real_output = run_program_to(code, self.__front_code, self.__end_code, entry_code, timeout)
        return real_output, exp_output

