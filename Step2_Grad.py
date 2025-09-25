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
import cv2
from MultimodalResNet import MultimodalModel

# --- 配置参数 ---
CSV_PATH = 'datasets/features.csv'
IMAGES_DIR = 'datasets'
BATCH_SIZE = 64
SEED = 42
EMBED_DIM = 512

SAVE_MODEL_DIR = './model'
SAVE_REPORT_DIR = './report'
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
            print(f"警告：文件未找到，跳过：{img_path}")
            return None, None, None
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(features), label

# --- 2. 加载数据并准备进行预测 ---
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

# --- 3. 绘图和保存热图的函数 ---
def visualize_grad_cams(model, val_dataset, save_dir, num_samples_per_class=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 尝试不同的中间层名称，看看哪个能生成更好的热图
    # 在尝试不同的名称时，一次只取消一个的注释
    # target_layer_name = 'image_feature_extractor.4.1.conv2' # ResNet layer1 结束
    # target_layer_name = 'image_feature_extractor.5.1.conv2' # ResNet layer2 结束
    target_layer_name = 'image_feature_extractor.6.1.conv2' # ResNet layer3 结束

    final_conv_layer = None
    for name, module in model.named_modules():
        if name == target_layer_name:
            final_conv_layer = module
            break

    if final_conv_layer is None:
        print(f"警告：未找到名称为 '{target_layer_name}' 的目标层，无法执行 Grad-CAM。")
        print("请手动检查您的模型结构以确认正确的名称。")
        return

    gradients = {}
    activations = {}

    def save_gradient(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]

    def save_activation(module, input, output):
        activations['value'] = output

    final_conv_layer.register_forward_hook(save_activation)
    final_conv_layer.register_full_backward_hook(save_gradient)

    unique_labels = sorted(list(val_dataset.dataframe['label_encoded'].unique()))
    label_map = val_dataset.label_map_dict

    grad_cam_root_dir = os.path.join(save_dir, 'grad_cam_visualizations')
    os.makedirs(grad_cam_root_dir, exist_ok=True)
    print(f"\nGrad-CAM heatmaps will be saved to subdirectories under: {grad_cam_root_dir}")

    samples_per_class = {label: [] for label in unique_labels}
    for i in range(len(val_dataset)):
        img, features, label = val_dataset[i]
        if img is not None:
            if len(samples_per_class[label]) < num_samples_per_class:
                samples_per_class[label].append({'img': img, 'features': features, 'label': label})

    for label, samples in samples_per_class.items():
        if not samples: continue

        class_name = label_map[label].replace('/', '_').replace('\\', '_')
        class_dir = os.path.join(grad_cam_root_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        print(f"Saving Grad-CAM images for class '{class_name}' to: {class_dir}")

        for i, sample in enumerate(samples):
            image_tensor = sample['img'].to(device)
            features_tensor = sample['features'].to(device)

            model.eval()
            output = model(image_tensor.unsqueeze(0), features_tensor.unsqueeze(0))

            model.zero_grad()
            one_hot_output = torch.zeros_like(output)
            one_hot_output[0][label] = 1
            output.backward(gradient=one_hot_output, retain_graph=True)

            grads_val = gradients['value'].cpu().data.numpy()[0]
            activations_val = activations['value'].cpu().data.numpy()[0]

            weights = np.mean(grads_val, axis=(1, 2))
            cam = np.zeros(activations_val.shape[1:], dtype=np.float32)
            for j, w in enumerate(weights):
                cam += w * activations_val[j, :, :]

            cam = np.maximum(cam, 0)
            if np.max(cam) > 0:
                cam = cam / np.max(cam)
            cam = cv2.resize(cam, (image_tensor.shape[2], image_tensor.shape[1]))

            original_img = image_tensor.permute(1, 2, 0).cpu().numpy()
            original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
            overlaid_img = heatmap_rgb * 0.5 + original_img * 0.5

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(original_img)
            axes[0].set_title(f'Original Image {i+1}')
            axes[0].axis('off')

            axes[1].imshow(overlaid_img)
            axes[1].set_title(f'Grad-CAM Heatmap {i+1}')
            axes[1].axis('off')

            plt.tight_layout()
            filename = f'grad_cam_sample_{i+1}.png'
            file_path = os.path.join(class_dir, filename)
            plt.savefig(file_path)
            print(f"Saved Grad-CAM visualization for sample {i+1} to: {file_path}")
            plt.close(fig)

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

                visualize_grad_cams(model, val_dataset, SAVE_REPORT_DIR)