import random
import numpy as np
from static.structs import TestSeed
from utils.other import ConfigManager
from typing import List, Dict, Any


class TestFuzzer:
    """ 
    生成测试用例的类，包括生成种子、生成测试用例等

    :param config_manager: ConfigManager，配置管理器    
    """
    default_seed = {
        "int": TestSeed("default_int", "int", [1]),
        "float": TestSeed("default_float", "float", [1.0]),
        "str": TestSeed("default_str", "str", ["hello"]),
        "bool": TestSeed("default_bool", "bool", [True]),
        "list": TestSeed("default_list", "list", [[1, 2, 3]]),
        "tuple": TestSeed("default_tuple", "tuple", [(1, 2, 3)]),
        "dict": TestSeed("default_dict", "dict", [{"a": 1, "b": 2}]),
        "set": TestSeed("default_set", "set", [{1, 2, 3}])
    }

    default_constants = {
        "int": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "float": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "str": ["123", "2312", "453432", "5342", "9876543"],
        "bool": [True, False],
        "list": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "tuple": [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
        "dict": [{"a": 1, "b": 2}, {"c": 3, "d": 4}, {"e": 5, "f": 6}],
        "set": [{1, 2, 3}, {4, 5, 6}, {7, 8, 9}]
    }

    def __init__(self, config_manager: ConfigManager = None):
        self.config_manager = config_manager
        self.config = config_manager.config if config_manager else {}

    def generate_multimodal_samples(
        self, constant_points, num_samples=10, stddev=0.5, weight=None
    ):
        """
        从一组常数点生成多峰分布并采样点。 
        
        :param constant_points: List[int/float]，常数点作为多峰分布的中心。
        :param num_samples: int，采样的总数。
        :param stddev: float，每个高斯分布的标准差。
        :param weight: List[float]，每个常数点的权重，默认为均匀分布。
        :return: List[float]，采样的测试用例点。
        """
        if not constant_points:
            raise ValueError("constant_points 不能为空")

        # 初始化权重
        if weight is None:
            weight = np.ones(len(constant_points)) / len(constant_points)
        if len(weight) != len(constant_points):
            raise ValueError("weight 的长度应与 constant_points 相等")

        # 确保权重归一化
        weight = np.array(weight) / sum(weight)

        # 采样：按权重随机选择一个常数点的分布
        samples = []
        for _ in range(num_samples):
            # 根据权重随机选择一个峰
            chosen_idx = np.random.choice(len(constant_points), p=weight)
            # 从对应高斯分布中采样
            mean = constant_points[chosen_idx]
            sample = np.random.normal(loc=mean, scale=stddev)
            samples.append(sample)

        return samples

    def fuzz(
        self, node, test_seed: TestSeed, constants: Dict[type, List[Any]] = None,
        tc_num_fuzzed: int = 10, stddev: float = 2
    ):
        """ 模糊测试生成测试用例，包括随机值和常量值。
        :param test_seed: 测试用例种子；
        :param constants: 代码中出现的常数，也作为峰值点；
        :param fuzz_num: 生成测试用例的数量；
        :param stddev: 标准差，越大生成的数据越偏离峰值，越小越集中于峰值；
        :return len of test_seed.vals: int，生成的测试用例数量；
        """
        if len(test_seed.vals) == tc_num_fuzzed:  # 防止重复fuzz
            return len(test_seed.vals)

        # 对于没有常数或没有种子的情况，使用默认值生成
        if (not constants) or (test_seed._type not in constants):
            constants = self.default_constants
        if not test_seed.vals:
            print(f"{test_seed.param_name} 没有种子，使用默认值生成")
            test_seed.vals = self.default_seed[test_seed._type].vals

        fuzz_num=tc_num_fuzzed - len(test_seed.vals)  # 实际需要再fuzz的tc数量

        # try:  
        if test_seed._type == "int":
            # 随机填充"UNCOVERED"占位符（原先测试用例没覆盖）
            for idx, val in enumerate(test_seed.vals):
                if val == "UNCOVERED":
                    test_seed.vals[idx] = random.choice(
                    self.default_constants["int"]
                )
            if isinstance(test_seed.vals[0], str):
                test_seed._type = "str"
                test_seed.vals += [f"{random.randint(0, 100)}" for _ in range(fuzz_num)]
                test_seed.fuzzed = True
                # 验证fuzz后的种子数量是否正确
                assert (
                    len(test_seed.vals) == tc_num_fuzzed
                ), "fuzz num not equal to expected error"
                return len(test_seed.vals)

            # 原有值 + 常量值，作为多峰分布中的峰值点
            peaks = list(
                test_seed.vals + list(constants[test_seed._type]),
            )
            # 采用多峰分布采样值
            test_seed.vals.extend(
                map(
                    int,
                    self.generate_multimodal_samples(
                        peaks, num_samples=fuzz_num, stddev=stddev
                    ),
                )
            )
        elif test_seed._type == "float":
            # 随机填充"UNCOVERED"占位符（原先测试用例没覆盖）
            for idx, val in enumerate(test_seed.vals):
                if val == "UNCOVERED":
                    test_seed.vals[idx] = random.choice(
                    self.default_constants["float"]
                )
            # 原有值 + 常量值，作为多峰分布中的峰值点
            peaks = test_seed.vals + list(constants[test_seed._type])
            # 采用多峰分布采样值
            test_seed.vals.extend(
                self.generate_multimodal_samples(
                    peaks,
                    # num_samples=self.config["fuzz_num"],
                    num_samples=fuzz_num,
                    stddev=stddev
                )
            )
        elif test_seed._type == "str":
            # 随机填充"UNCOVERED"占位符（原先测试用例没覆盖）
            for idx, val in enumerate(test_seed.vals):
                if val == "UNCOVERED":
                    test_seed.vals[idx] = random.choice(
                    self.default_constants["str"]
                )

            test_seed.vals += [f"str_{random.randint(0, 100)}" for _ in range(fuzz_num)]
        elif test_seed._type == "bool":
            for idx, val in enumerate(test_seed.vals):
                if val == "UNCOVERED":
                    test_seed.vals[idx] = random.choice(
                    self.default_constants["bool"]
                )
            test_seed.vals += [random.choice([True, False]) for _ in range(fuzz_num)]
        elif test_seed._type == "list":
            # 随机一个数组长度
            # 采样fuzz_num次
            for idx, val in enumerate(test_seed.vals):
                if val == "UNCOVERED":
                    test_seed.vals[idx] = random.choice(
                    self.default_constants["list"]
                )
            test_seed.vals += [
                random.choice(self.default_constants["list"])
                for _ in range(fuzz_num)
            ]
        elif test_seed._type == "tuple":
            for idx, val in enumerate(test_seed.vals):
                if val == "UNCOVERED":
                    test_seed.vals[idx] = random.choice(
                    self.default_constants["tuple"]
                )
            test_seed.vals += [
                random.choice(self.default_constants["tuple"])
                for _ in range(fuzz_num)
            ]
        else:
            raise ValueError(f"Unsupported type: {test_seed._type}")
        test_seed.fuzzed = True
        # 验证fuzz后的种子数量是否正确
        assert (
            len(test_seed.vals) == tc_num_fuzzed
        ), "fuzz num not equal to expected error"
        return len(test_seed.vals)
