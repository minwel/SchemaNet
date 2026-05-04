import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from classify.dynamic_info import get_dynamics

# DYNAMIC_MAX_LEN = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SampleDataset(Dataset):
    def __init__(self, fnames, texts, labels, code_src, dynamic_max_len):
        self.fnames = fnames
        self.texts = texts
        self.labels = labels
        self.code_src = code_src
        self.dynamic_max_len = dynamic_max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        dynamic_encodes = get_dynamics(
            os.path.join(self.code_src, self.fnames[idx])
        )

        # 判断长度并进行截断或填充
        if len(dynamic_encodes) > self.dynamic_max_len:
            dynamic_encodes = dynamic_encodes[:self.dynamic_max_len]  # 截断
        else:
            dynamic_encodes = np.pad(
                dynamic_encodes,
                (0, self.dynamic_max_len - len(dynamic_encodes)),  # 填充
                "constant",
            )

        dynamic_encodes = dynamic_encodes.astype(np.float32)  # 转换为float32类型

        text = self.texts[idx]
        label = self.labels[idx]
        
        return text, dynamic_encodes, label


class CodeDataLoader:
    def __init__(self, dataset_path, code_src, batch_size, split_ratio, dynamic_max_len):
        self.batch_size = batch_size
        self.code_src = code_src
        self.dynamic_max_len = dynamic_max_len   
        # 加载数据
        df = pd.read_excel(dataset_path)
        # 提取
        self.fname_list = df['fname'].tolist()
        self.code_list = df['code'].tolist()
        self.label_list = df['label'].tolist()
        # 分割
        self.split_set = train_test_split(
            self.fname_list, self.code_list, self.label_list, 
            train_size=split_ratio, random_state=42
        )  # train_fname, test_fname, train_code, test_code, train_label, test_label
        print(f"data src: {dataset_path}")
        print(f"训练集大小: {len(self.split_set[0])}")
        print(f"测试集大小: {len(self.split_set[1])}")

    def get4cluster(self, ):
        """ 用于聚类goal/plan的 """
        return self.split_set[:4]
    
    def get4train_batch(self, ):
        """ 用于分类任务训练 """
        (
            train_fname,
            temp_fname,
            train_codes,
            temp_codes,
            train_labels,
            temp_labels,
        ) = self.split_set

        (
            train_sub_fname,
            val_sub_fname,
            train_sub_codes,
            val_sub_codes,
            train_sub_labels,
            val_sub_labels,
        ) = train_test_split(
            train_fname,
            train_codes,
            train_labels,
            train_size=0.9,
            random_state=42,
        )

        test_fname = temp_fname
        test_codes = temp_codes
        test_labels = temp_labels

        # 创建训练集、测试集的数据加载器
        train_dataset = SampleDataset(train_sub_fname, train_sub_codes, train_sub_labels, self.code_src, self.dynamic_max_len)
        val_dataset = SampleDataset(val_sub_fname, val_sub_codes, val_sub_labels, self.code_src, self.dynamic_max_len)
        test_dataset = SampleDataset(test_fname, test_codes, test_labels, self.code_src, self.dynamic_max_len)

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, 
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, 
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, 
        )

        return train_dataloader, val_dataloader, test_dataloader
