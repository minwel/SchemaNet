import os
import xml.etree.ElementTree as ET

class TestCasesLoader:
    """
        Load test cases from folder, in which xml-style files exists.\n
        Return: 
         - Dict[str(id):(list[input], list[output])]
    """
    
    @classmethod
    def xml2json(cls, xml_path: str) -> dict:
        # 解析 XML 文件
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 递归函数，将 Element 对象转换为字典
        def element_to_dict(element):
            result = {}
            if element.attrib:
                result.update(element.attrib)
            if element.text:
                result[element.tag] = element.text
            for child in element:
                child_dict = element_to_dict(child)
                result.setdefault(child.tag, []).append(child_dict)
            return result

        # 将根节点转换为字典
        json_data = element_to_dict(root)
        return json_data
    
    @classmethod
    def parse_dict(cls, dic: dict) -> tuple:
        count = int(dic['count'])
        inputs = []
        outputs = []
        for i in range(count):
            case_name = 'testData' + str(i + 1)
            inputs.append(dic[case_name][0]['input'][0]['input'].strip())
            outputs.append(dic[case_name][0]['output'][0]['output'].strip())
        return inputs, outputs

    @classmethod
    def get_test_cases(cls, test_cases_folder_path: str) -> dict:
        test_cases_name = os.listdir(test_cases_folder_path)
        return {
            os.path.splitext(name)[0][-4:]:
            cls.parse_dict(
                cls.xml2json(
                    os.path.join(test_cases_folder_path, name)
                )
            ) for name in test_cases_name if name.endswith('.xml')
        }


