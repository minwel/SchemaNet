import ast
import numpy as np
import torch
import torch.nn as nn
from SchemaMiner import get_encodes_onehot
from trees.tree import ASTNode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PlanNBertNDynamicsClassifier(nn.Module):
    def __init__(self, encoder_bert, num_classes, plans, dynamic_max_len):
        """
        :vocab_size:
        :embed_dim:
        """
        super(PlanNBertNDynamicsClassifier, self).__init__()
        self.plans = plans
        self.encoder = encoder_bert
        self.linear = nn.Linear(len(self.plans), 128)  # 对plan进行线性层升维
        self.dropout = nn.Dropout(p=0.5)
        # fc：bert_hidden + plan_len + dynmcs_len -> num_classes
        self.fc = nn.Linear(self.encoder.hidden_size + 128 + dynamic_max_len, num_classes)  # 完整版本

        print(f"PlanNBertNDynamicsClassifier: dynamic_max_len: {dynamic_max_len}")

    def forward(self, texts, dynamic_embedded):
        # bert 编码
        bert_embedded = self.encoder.single_encoding(texts)  # [32, 768]

        # plan 编码
        def tmp(text):
            plan_encodes = get_encodes_onehot(
                plans=self.plans,
                bug_tree=ASTNode(ast.parse(text), is_correct=False), 
            )
            return plan_encodes        
        plan_embedded = np.array([tmp(t) for t in texts])  # [32, len(self.plans)]
        plan_embedded = torch.tensor(plan_embedded, dtype=torch.float32).to(device)
        plan_embedded = self.linear(plan_embedded)  # [32, 128]  线性层升维

        # concat
        combined_vector = torch.cat((bert_embedded, plan_embedded, dynamic_embedded), dim=-1)  # 完整版

        # 在送入分类器前应用dropout
        combined_vector_for_cls = self.dropout(combined_vector)

        # cls
        out = self.fc(combined_vector_for_cls)
        
        return out, combined_vector


