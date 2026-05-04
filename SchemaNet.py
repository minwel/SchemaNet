import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.other import ConfigManager
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from containers import Application
from encoder.codebert import CodeBert
from dependency_injector.wiring import Provide, inject
from classify.model import PlanNBertNDynamicsClassifier
from SchemaMiner import get_clustered_res
from utils.file import check_paths
from sklearn.metrics import classification_report, accuracy_score
from utils.codedataloader import CodeDataLoader
from utils.other import print_split_line

torchvision.disable_beta_transforms_warning()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置
num_classes = 6 
batch_size = 32
epochs = 100
learning_rate = 2e-3
best_accuracy = 0.0
no_improvement_count = 0
patience = 10  # 连续没有提升的epoch阈值
model_select = 'plan_bert_dynmcs'

dynamic_max_len = 128
static_max_len = 128
encoder_codebert = None
prob_id = 3004
saved_model_path = str(
    PROJECT_ROOT / "tmp" / "model" / str(prob_id) / f"best_model_{model_select}.pth"
)  # 最佳模型保存地址

cls_tree = None
plans = []
model = None
criterion = None
optimizer = None
train_dataloader = None
val_dataloader = None
test_dataloader = None

@inject
def init_model():
    global model, criterion, optimizer, encoder_codebert

    model = PlanNBertNDynamicsClassifier(
        encoder_codebert, num_classes, plans, dynamic_max_len
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 加载已有模型
    if os.path.exists(saved_model_path):
        print(f"Loading saved model from {saved_model_path}")
        model.load_state_dict(torch.load(saved_model_path))
    else:
        print(f"No saved model found at {saved_model_path}, training a new model...")


def train():
    global best_accuracy, no_improvement_count
    print_split_line('开始训练')
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        for batch_idx, (text, dynmcs, target) in enumerate(train_dataloader):
            dynmcs = dynmcs.to(device)
            target = target.to(device)
            # Forward pass
            output, _ = model(text, dynmcs)
            # 反向传播和优化
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 1 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(train_dataloader)}], Loss: {loss.item():.4f}')

        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for text, dynmcs, target in val_dataloader:
                dynmcs = dynmcs.to(device)
                target = target.to(device)
                # Forward pass
                output, _ = model(text, dynmcs)

                # 预测类别（取每行 logits 最大值对应的索引）
                preds = torch.argmax(output, dim=1)

                # 计算准确率
                correct += (preds == target).sum().item()
                total += target.size(0)

        val_accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{epochs}], Validation Accuracy: {val_accuracy:.2f}%')

        # 如果当前准确率高于最佳准确率，则更新最佳准确率并保存模型权重
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), saved_model_path)
            print(f'Model with best validation accuracy ({val_accuracy:.2f}%) saved!')
            no_improvement_count = 0  # 重置计数器
        else:
            no_improvement_count += 1  # 增加计数器

        if no_improvement_count >= patience:
            print(f'Stopping training after {patience} epochs without improvement.')
            break


def test():
    # 评估模型
    model.load_state_dict(torch.load(saved_model_path)) 
    model.eval()
    predictions = [] 
    solutions = []
    print_split_line('开始评估')
    print(32 * len(test_dataloader))
    with torch.no_grad():
        correct = 0
        total = 0
        for text, dynmcs, target in test_dataloader:
            dynmcs = dynmcs.to(device)
            target = target.to(device)
            # Forward pass
            output, _ = model(text, dynmcs)

            solutions.extend(map(int, target))
            preds = torch.argmax(output, dim=1)

            # 计算准确率
            correct += (preds == target).sum().item()
            total += target.size(0)

            predictions.extend(preds.detach().cpu().numpy())

        print(
            f"Accuracy of the model on the {len(test_dataloader.dataset)} "
            f"samples: {100 * accuracy_score(solutions, predictions):.2f}%"
        )
        print(f'分类报告:')
        print(classification_report(solutions, predictions, digits=4))


def main(
    config_manager: ConfigManager = Provide[Application.config_manager],
    dataloader: CodeDataLoader = Provide[Application.dataloader],
):
    global cls_tree, train_dataloader, val_dataloader, test_dataloader, plans
    global model_select, prob_id, num_classes, saved_model_path
    global patience, learning_rate, dynamic_max_len, static_max_len, encoder_codebert

    config = config_manager.config

    parser = argparse.ArgumentParser()
    parser.add_argument("-pr", "--prob", type=str, default="3004", help="problem id")
    parser.add_argument("-c", "--cls_num", type=int, default=6, help="class num")
    parser.add_argument("-pt", "--patience", type=int, default=10, help="patience")
    parser.add_argument("-lr", "--learning_rate", type=float, default=2e-3, help="learning rate")
    parser.add_argument("-sp", "--split_ratio", type=float, default=0.6, help="split ratio for train/test")
    args = parser.parse_args()

    prob_id = args.prob
    model_select = "plan_bert_dynmcs"
    num_classes = args.cls_num
    patience = args.patience
    learning_rate = args.learning_rate
    saved_model_path = str(
        PROJECT_ROOT / "tmp" / "model" / str(prob_id) / f"best_model_{model_select}.pth"
    )

    dynamic_max_len = 128
    static_max_len = 128
    encoder_codebert = CodeBert(trees=None, static_max_len=static_max_len)

    config['code_from'] = args.prob
    config['code_src'] = "data/test/{code_from}"
    config['true_code_path'] = "data/test/{code_from}.xlsx"


    # config
    config['split_ratio'] = args.split_ratio

    # 重置已由containers初始化的对象
    config_manager.__init__(config)
    dataloader.__init__(
        dataset_path=config["true_code_path"],
        code_src=config["code_src"],
        batch_size=batch_size,
        split_ratio=config["split_ratio"],
        dynamic_max_len=dynamic_max_len,
    )

    # 创建模型保存目录
    check_paths(os.path.dirname(saved_model_path))

    # 获取cls_tree
    cls_tree = get_clustered_res('plan' in model_select)
    # 获取所有的plans
    plans = [trees[0] for _, _plans in cls_tree.items() for _, trees in _plans.items()]

    print_split_line('开始分类任务')
    print(f"Plan序列维度：{len(plans)}")
    print(f"分类模型：{model_select}")
    print(f"分类数：{num_classes}")
    print(f"问题编号：{prob_id}")
    print(f"patience：{patience}")
    print(f"学习率：{learning_rate}")

    init_model()

    dataloader.__init__(
        dataset_path=config["true_code_path"],
        code_src=config["code_src"],
        batch_size=batch_size,
        split_ratio=config["split_ratio"],
        dynamic_max_len=dynamic_max_len,
    )

    train_dataloader, val_dataloader, test_dataloader = dataloader.get4train_batch()

    train()
    test()


if __name__ == '__main__':
    application = Application()
    application.init_resources()
    application.wire(modules=[__name__, "SchemaMiner", ])
    main()
