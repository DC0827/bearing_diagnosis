import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE

# ⚠️ 注意：这里假设你的 MultimodalModel 类定义在 MultimodalResNet.py 中，并且其 forward 方法没有 return_features 参数
from MultimodalResNet import MultimodalModel

# --- 配置参数 ---
CSV_PATH = 'datasets/features.csv'
IMAGES_DIR = 'datasets'
BATCH_SIZE = 128
SEED = 42
EMBED_DIM = 512

SAVE_MODEL_DIR = './model'
SAVE_REPORT_DIR = './report128'
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

# --- 1. 自定义数据集 (保持不变) ---
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
            print(f"警告：文件未找到，跳过：{img_path}")
            return None, None, None
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(features), label

# --- 2. 加载数据并准备进行预测 (保持不变) ---
def prepare_data_for_inference():
    df = pd.read_csv(CSV_PATH)
    df['file_id'] = df['file_index'].apply(lambda x: x.split('_')[0])
    unique_files = df['file_id'].unique()
    file_labels = df.drop_duplicates('file_id').set_index('file_id')['label']
    file_labels_encoded, _ = pd.factorize(file_labels)
    _, val_files = train_test_split(
        unique_files,
        test_size=0.2,
        random_state=SEED,
        stratify=file_labels_encoded
    )
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

    scaler_path = os.path.join(SAVE_MODEL_DIR, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        print(f"错误：标准化器文件未找到：{scaler_path}")
        return None, None, None, None, None
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"已加载标准化器: {scaler_path}")
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])

    num_classes = len(df['label'].unique())
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = ImageDataset(val_df, IMAGES_DIR, feature_cols, transform)

    def collate_fn(batch):
        batch = list(filter(lambda x: x[0] is not None, batch))
        if len(batch) == 0:
            return None, None, None
        return torch.utils.data.dataloader.default_collate(batch)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    return val_loader, num_classes, val_dataset.label_map_dict, len(feature_cols), val_dataset

# --- 3. 编写 t-SNE 可视化函数 ---
def visualize_tsne_of_features(model, data_loader, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 专门用于存储特征的字典
    features_dict = {}

    # 定义钩子函数
    def get_features_hook(module, input, output):
        # 这个钩子函数将中间层的输出保存到 features_dict 中
        features_dict['fused_features'] = output.cpu().detach()

    # 注册钩子
    # 找到融合层并注册前向传播钩子
    hook = model.fusion_layer.register_forward_hook(get_features_hook)

    all_features = []
    all_labels = []

    print("\n开始从模型中提取特征...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="提取特征"):
            if batch[0] is None:
                continue
            images, tabs, labels = batch
            images = images.to(device)
            tabs = tabs.to(device)

            # 调用模型进行前向传播
            # 钩子会自动捕获 model.fusion_layer 的输出
            _ = model(images, tabs)

            # 从钩子中获取保存的特征
            features = features_dict['fused_features']
            all_features.append(features.numpy())
            all_labels.append(labels.numpy())

    # 移除钩子以清理资源
    hook.remove()

    # 合并所有特征和标签
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 4. 应用 t-SNE 降维
    print("应用 t-SNE 降维...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=SEED)
    tsne_results = tsne.fit_transform(all_features)
    print("t-SNE 降维完成。")

    # 5. 绘制散点图并保存
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(all_labels)
    for label in unique_labels:
        indices = all_labels == label
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                    label=data_loader.dataset.label_map_dict[label],
                    alpha=0.7)

    plt.title('t-SNE Visualization of Fused Features (Using Hooks)', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'tsne_visualization_hook.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"t-SNE 可视化图已保存到：{save_path}")
    plt.show()

# --- 主执行 ---
if __name__ == "__main__":
    set_seed(SEED)
    if not os.path.exists(CSV_PATH) or not os.path.exists(IMAGES_DIR):
        print(f"错误：所需文件 '{CSV_PATH}' 或文件夹 '{IMAGES_DIR}' 不存在。")
    else:
        val_loader, num_classes, label_map, num_features, val_dataset = prepare_data_for_inference()
        if val_loader is None:
            print("数据准备失败，请检查文件和文件夹路径。")
        else:
            model = MultimodalModel(num_classes, num_features, EMBED_DIM)
            model_path = os.path.join(SAVE_MODEL_DIR, 'best_model.pth')
            if not os.path.exists(model_path):
                print(f"错误：模型文件未找到：{model_path}")
            else:
                model.load_state_dict(torch.load(model_path))
                print(f"已加载模型: {model_path}")

                # 调用新的 t-SNE 可视化函数
                visualize_tsne_of_features(model, val_loader, SAVE_REPORT_DIR)