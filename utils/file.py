import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


def check_paths(*file_paths: str):
    for fp in file_paths:
        os.makedirs(fp, exist_ok=True)

def read_yaml(fp: str):
    import yaml
    with open(fp, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data


