import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class MultimodalModel(nn.Module):
    def __init__(self, num_classes, num_features, embed_dim=512):
        super().__init__()
        
        # -------- 图像特征提取 --------
        resnet = models.resnet50(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # 使用ResNet18
        self.image_feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # (B, 2048, 7, 7)

        # -------- 表格特征投影 + 门控 --------
        self.csv_mlp = nn.Sequential(
            nn.Linear(num_features, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
        self.csv_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()  # 使用ReLU替代Sigmoid
        )

        # -------- 图像特征投影 --------
        self.image_projection = nn.Sequential(
            nn.Linear(512, embed_dim),  # 将图像特征投影到embed_dim维度
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )

        # -------- 融合层 --------
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),  # 图像特征和表格特征经过投影后拼接
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # -------- 分类头 --------
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, img_input, tab_input):
        B = img_input.size(0)

        # ===== 图像路径 =====
        img_features = self.image_feature_extractor(img_input)   # (B, 2048, 7, 7)
        img_features_pooled = F.adaptive_avg_pool2d(img_features, (1, 1))  # (B, 2048, 1, 1)
        image_features_flattened = torch.flatten(img_features_pooled, 1)   # (B, 2048)
        image_features_projected = self.image_projection(image_features_flattened)  # (B, embed_dim)

        # ===== CSV 路径 (MLP + 门控) =====
        csv_emb = self.csv_mlp(tab_input)  # (B, 512)
        gate = self.csv_gate(csv_emb)
        csv_features_scaled = csv_emb * gate  # (B, 512)

        # ===== 融合 =====
        combined_features = torch.cat((image_features_projected, csv_features_scaled), dim=1)  # (B, embed_dim * 2)

        # ===== 融合层 =====
        fused_features = self.fusion_layer(combined_features)

        # ===== 分类 =====
        outputs = self.classifier(fused_features)
        return outputs
