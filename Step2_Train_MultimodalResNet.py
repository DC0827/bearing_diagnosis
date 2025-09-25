import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler, label_binarize
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns # 导入 seaborn 库
from MultimodalResNet import MultimodalModel # 自己的多模态模型

# --- 配置参数 ---
CSV_PATH = 'datasets/features.csv'
IMAGES_DIR = 'datasets'
NUM_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
SEED = 42
EMBED_DIM = 512
NUM_HEADS = 8
PATIENCE = 10  # 早停的耐心值

SAVE_MODEL_DIR = './model'
SAVE_REPORT_DIR = './report'
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
os.makedirs(SAVE_REPORT_DIR, exist_ok=True)

# --- 0. 设置随机种子 ---
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- 1. 自定义数据集 ---
class ImageDataset(Dataset):
    def __init__(self, dataframe, img_dir, feature_cols, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.feature_cols = feature_cols
        self.transform = transform

        labels_encoded, label_map = pd.factorize(self.dataframe['label'])
        self.label_map_dict = {i: label for i, label in enumerate(label_map)}
        self.dataframe['label_encoded'] = labels_encoded

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        file_index = row['file_index']
        label = row['label_encoded']
        features = row[self.feature_cols].values.astype(np.float32)
        original_label = self.label_map_dict[label]
        img_path = os.path.join(self.img_dir, original_label, f"{file_index}.png")

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"警告：文件未找到,跳过：{img_path}")
            return None, None, None

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(features), label

# --- 2. 加载和准备数据（标准化并保存 scaler） ---
def prepare_data():
    df = pd.read_csv(CSV_PATH)
    df['file_id'] = df['file_index'].apply(lambda x: x.split('_')[0])

    unique_files = df['file_id'].unique()
    file_labels = df.drop_duplicates('file_id').set_index('file_id')['label']
    file_labels_encoded, _ = pd.factorize(file_labels)

    train_files, val_files = train_test_split(
        unique_files,
        test_size=0.2,
        random_state=SEED,
        stratify=file_labels_encoded
    )

    train_df = df[df['file_id'].isin(train_files)].copy()
    val_df = df[df['file_id'].isin(val_files)].copy()

    feature_cols = [
    "均值 (T1)", "均方根 (T2)", "峰值 (T3)", "峰-峰值 (T4)", "峭度 (T5)", "波形指标 (T6)", 
    "峰值指标 (T7)", "脉冲指标 (T8)", "裕度指标 (T9)", "峭度指标 (T10)", "偏度 (T11)", 
    "裂隙因子(T12)", "过零率(T13)", "BPFO谐波幅值(f1)", "BPFI谐波幅值(f2)", "BSF谐波幅值(f3)", 
    "转频幅值(f4)", "频谱质心(f5)", "谱平坦度(f6)", "谱带宽(f7)", "高频能量占比(f8)", 
    "高频峭度(f9)", "小波包特征1", "小波包特征2", "小波包特征3", "小波包特征4", 
    "小波包特征5", "小波包特征6", "小波包特征7", "小波包特征8", "包络最大值(e1)", 
    "包络最小值(e2)", "包络均值(e3)", "包络标准差(e4)", "包络峰度(e5)", "包络偏度(e6)", 
    "包络频谱峰值(e7)", "包络频谱均值(e8)", "幅值熵(s1)", "谱熵(s1)"
    ]

    # 标准化
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])

    # 保存 scaler
    scaler_path = os.path.join(SAVE_MODEL_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"已保存标准化器到: {scaler_path}")

    num_classes = len(df['label'].unique())
    print(f"检测到的类别数量：{num_classes}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageDataset(train_df, IMAGES_DIR, feature_cols, transform)
    val_dataset = ImageDataset(val_df, IMAGES_DIR, feature_cols, transform)

    def collate_fn(batch):
        batch = list(filter(lambda x: x[0] is not None, batch))
        if len(batch) == 0:
            return None, None, None
        return torch.utils.data.dataloader.default_collate(batch)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, num_classes, val_dataset.label_map_dict, len(feature_cols)

# --- 3. 训练和验证循环 ---
def train_model(model, train_loader, val_loader, num_epochs, label_map):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备：{device}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

     # 使用 StepLR 实现学习率衰减，每50个epoch将学习率减少一半
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    metrics_history = []
    best_val_f1_score = -1.0 # 使用F1分数作为最佳模型保存依据
    patience_counter = 10 # 早停计数器

    # 存储最佳模型的验证数据
    best_val_labels = None
    best_val_outputs = None
    best_val_report = None
    best_epoch = 0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [训练中]")
        for inputs, features, labels in train_bar:
            if inputs is None or labels is None:
                continue
            inputs, features, labels = inputs.to(device), features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix(loss=loss.item())
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | 训练损失: {epoch_loss:.4f}")

        # 验证阶段
        model.eval()
        val_loss, val_corrects, total_val = 0.0, 0, 0
        all_preds, all_labels, all_outputs = [], [], []
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [验证中]")
        for inputs, features, labels in val_bar:
            if inputs is None or labels is None:
                continue
            inputs, features, labels = inputs.to(device), features.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs, features)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
            total_val += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

        val_epoch_loss = val_loss / total_val
        val_acc = val_corrects.double() / total_val

        target_names = [label_map[i] for i in range(len(label_map))]
        report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True, zero_division=0)
        val_f1_score = report['macro avg']['f1-score']

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | 验证损失: {val_epoch_loss:.4f} | 验证准确率: {val_acc:.4f} | 验证 F1-score: {val_f1_score:.4f}")

        # 早停和模型保存逻辑：基于F1分数
        if val_f1_score > best_val_f1_score:
            best_val_f1_score = val_f1_score
            patience_counter = 0
            model_save_path = os.path.join(SAVE_MODEL_DIR, 'best_model.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"F1分数上升,保存最优模型到：{model_save_path}")

            # 保存最佳模型的验证数据
            best_val_labels = np.array(all_labels)
            best_val_outputs = np.array(all_outputs)
            best_val_report = report
            best_epoch = epoch + 1
        else:
            patience_counter += 1
            print(f"F1分数未上升,早停计数器: {patience_counter}/{PATIENCE}")

        print("\n--- 验证集分类报告 ---")
        print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
        print("---------------------\n")

        metrics_history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'val_loss': val_epoch_loss,
            'val_accuracy': val_acc.item(),
            'val_f1_score': val_f1_score,
            'classification_report': report
        })

        if patience_counter >= PATIENCE:
            print(f"F1分数连续 {PATIENCE} 次未上升,停止训练。")
            break

        # 更新学习率
        scheduler.step()

    print("\n训练完成。")
    json_save_path = os.path.join(SAVE_REPORT_DIR, 'training_metrics.json')
    with open(json_save_path, 'w') as f:
        json.dump(metrics_history, f, indent=4)
    print(f"训练指标已保存到 {json_save_path} 文件中。")

    return metrics_history, best_val_labels, best_val_outputs, best_val_report, best_epoch

# --- 4. 绘图和保存报告的函数 ---
def plot_and_save_results(metrics_history, best_val_labels, best_val_outputs, best_val_report, best_epoch, num_classes, label_map):
    epochs = [m['epoch'] for m in metrics_history]

    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    train_losses = [m['train_loss'] for m in metrics_history]
    val_losses = [m['val_loss'] for m in metrics_history]
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    best_val_loss_at_best_epoch = metrics_history[best_epoch-1]['val_loss']
    plt.plot(best_epoch, best_val_loss_at_best_epoch, 'ro', label=f'Best Model Epoch: {best_epoch}')
    plt.text(best_epoch, best_val_loss_at_best_epoch, f"Loss: {best_val_loss_at_best_epoch:.4f}", ha='right', color='red')
    plt.title('Training and Validation Loss Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_fig_path = os.path.join(SAVE_REPORT_DIR, 'loss_curve.png')
    plt.savefig(loss_fig_path)
    print(f"损失曲线图已保存到: {loss_fig_path}")
    plt.show()

    # 绘制准确率曲线
    plt.figure(figsize=(12, 6))
    val_accuracies = [m['val_accuracy'] for m in metrics_history]
    plt.plot(epochs, val_accuracies, label='Overall Accuracy', color='green')
    best_acc = best_val_report['accuracy']
    plt.plot(best_epoch, best_acc, 'go', label=f'Best Accuracy: {best_acc:.4f}')
    plt.text(best_epoch, best_acc, f"Acc: {best_acc:.4f}", ha='right', color='green')
    plt.title('Overall Accuracy Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    acc_fig_path = os.path.join(SAVE_REPORT_DIR, 'accuracy_curve.png')
    plt.savefig(acc_fig_path)
    print(f"准确率曲线图已保存到: {acc_fig_path}")
    plt.show()

    # 绘制F1-score曲线
    plt.figure(figsize=(12, 6))
    val_f1_scores = [m['val_f1_score'] for m in metrics_history]
    plt.plot(epochs, val_f1_scores, label='Overall Macro F1-score', color='purple')
    best_f1 = best_val_report['macro avg']['f1-score']
    plt.plot(best_epoch, best_f1, 'mo', label=f'Best F1-score: {best_f1:.4f}')
    plt.text(best_epoch, best_f1, f"F1: {best_f1:.4f}", ha='left', color='purple')
    plt.title('Overall Macro F1-score Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1-score', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    f1_fig_path = os.path.join(SAVE_REPORT_DIR, 'f1_score_curve.png')
    plt.savefig(f1_fig_path)
    print(f"F1-score曲线图已保存到: {f1_fig_path}")
    plt.show()

    # 绘制整体ROC曲线和AUC值
    if best_val_labels is not None and best_val_outputs is not None:
        plt.figure(figsize=(6, 6))
        all_labels_one_hot = label_binarize(best_val_labels, classes=range(num_classes))

        # 绘制Micro-average ROC曲线
        fpr_micro, tpr_micro, _ = roc_curve(all_labels_one_hot.ravel(), best_val_outputs.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        plt.plot(fpr_micro, tpr_micro, label=f'Micro-average (AUC = {roc_auc_micro:.2f})', color='deeppink', linestyle=':', linewidth=4)

        # 绘制Macro-average ROC曲线
        all_aucs = []
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(all_labels_one_hot[:, i], best_val_outputs[:, i])
            all_aucs.append(auc(fpr, tpr))
        roc_auc_macro = np.mean(all_aucs)
        plt.plot(fpr_micro, tpr_micro, label=f'Macro-average (AUC = {roc_auc_macro:.2f})', color='navy', linestyle=':', linewidth=4)

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Overall ROC Curve and AUC (Best Model: Epoch {best_epoch})', fontsize=16)
        plt.legend(loc="lower right")
        plt.grid(True)
        roc_fig_path = os.path.join(SAVE_REPORT_DIR, 'overall_roc_curve.png')
        plt.savefig(roc_fig_path)
        print(f"整体ROC曲线图已保存到: {roc_fig_path}")
        plt.show()

    # 绘制混淆矩阵
    if best_val_labels is not None and best_val_outputs is not None:
        best_val_preds = np.argmax(best_val_outputs, axis=1)
        cm = confusion_matrix(best_val_labels, best_val_preds)
        target_names = [label_map[i] for i in range(num_classes)]

        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        cm_fig_path = os.path.join(SAVE_REPORT_DIR, 'confusion_matrix.png')
        plt.savefig(cm_fig_path)
        print(f"\nConfusion matrix plot saved to: {cm_fig_path}")
        plt.show()

    # 保存最佳模型的分类报告到txt文件
    report_str = "--- Best Model Classification Report ---\n\n"
    target_names = [label_map[i] for i in range(num_classes)]
    report_str += classification_report(best_val_labels, np.argmax(best_val_outputs, axis=1), target_names=target_names, zero_division=0)
    report_save_path = os.path.join(SAVE_REPORT_DIR, 'best_model_classification_report.txt')
    with open(report_save_path, 'w') as f:
        f.write(report_str)
    print(f"\n最佳模型的分类报告已保存到: {report_save_path}")

# --- 主执行 ---
if __name__ == "__main__":
    set_seed(SEED)
    if not os.path.exists(CSV_PATH) or not os.path.exists(IMAGES_DIR):
        print(f"错误：所需文件 '{CSV_PATH}' 或文件夹 '{IMAGES_DIR}' 不存在。")
    else:
        train_loader, val_loader, num_classes, label_map, num_features = prepare_data()
        if train_loader is None or val_loader is None:
            print("数据加载失败。请检查文件和文件夹路径。")
        else:
            model = MultimodalModel(num_classes, num_features, EMBED_DIM)
            metrics_history, best_labels, best_outputs, best_report, best_epoch = train_model(model, train_loader, val_loader, NUM_EPOCHS, label_map)

            # 训练完成后,绘制图表并保存报告
            plot_and_save_results(metrics_history, best_labels, best_outputs, best_report, best_epoch, num_classes, label_map)
# CUDA_VISIBLE_DEVICES=6 python Step4_Train_MultimodalResNet.py