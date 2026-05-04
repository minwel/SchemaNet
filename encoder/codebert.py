# type: ignore
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from overrides import override
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Union
from .base import SubtreeEncoder

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['HF_DATASETS_OFFLINE'] = '0'
os.environ['HF_HUB_OFFLINE'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CodeBert(SubtreeEncoder):
    model_path: str = "microsoft/codebert-base"
    batch_size: int = 32
    
    def __init__(self, trees: "Trees" = None, *args, **kwargs) -> None:
        super().__init__(trees, *args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_path, local_files_only=False,
        )
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.model_path, local_files_only=False
        ).to(device)
        if trees:
            self.subtrees = trees.trees
        self.hidden_size = self.model.config.hidden_size
        self.max_char_len = kwargs["static_max_len"]
        # print(f"CodeBert: {self.model_path}, hidden_size: {self.hidden_size}, max_char_len: {self.max_char_len}")

    def encode(self):
        self.model.eval()

        if not self.subtrees:
            print("warning! not initialized yet, only support single_encoding")
            return

        sbs_tokenized = self.tokenizer(
            [str(sb) for sb in self.subtrees],
            return_tensors="pt",
            max_length=self.max_char_len,
            padding="max_length",
            truncation=True,
        )
        input_ids = sbs_tokenized['input_ids']
        atten_vec = sbs_tokenized['attention_mask']
        dataloader = DataLoader(
            list(zip(input_ids, atten_vec, [sb.subtree_id for sb in self.subtrees])),
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.sbs_vecs = {} # {sb_id: sb_embedding}
        for batch in tqdm(dataloader, desc='codebert encoding'):
        # for batch in dataloader:
            input_ids, atten_vec, sb_ids = batch
            self.sbs_vecs.update(
                dict(
                    zip(
                        map(int, sb_ids),
                        self.model(
                            input_ids.to(device), attention_mask=atten_vec.to(device)
                        )["pooler_output"] 
                        .detach()
                        .cpu()
                        .numpy()
                    )
                )
            )
        torch.cuda.empty_cache()
        print(f"encoded dims: {len(self.sbs_vecs[0])}")

    @override
    def single_encoding(self, subtree: Union["ASTNode", str]) -> np.ndarray:
        """ 重写（实现接口方法） """

        if isinstance(subtree, (list, tuple)):
            inputs = self.tokenizer(
                subtree, padding="max_length", truncation=True, return_tensors="pt",
                max_length=self.max_char_len, 
            ).to(device)
        else:
            inputs = self.tokenizer(
                str(subtree), padding="max_length", truncation=True, return_tensors="pt",
                max_length=self.max_char_len, 
            ).to(device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # return outputs.pooler_output.detach().cpu().numpy().squeeze()  # plans聚类，单个编码
        return outputs.pooler_output  # 分类，批量推理
        # return self.sbs_vecs[subtree.subtree_id]

    @override
    def batch_encoding(self) -> List[np.ndarray]:
        return list(self.sbs_vecs.values())
