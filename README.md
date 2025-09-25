# 轴承故障诊断系统

该项目实现了一个基于**迁移学习**的轴承故障诊断系统，目的是通过源域数据训练模型，并通过迁移学习方法对目标域进行故障分类。我们使用了多种深度学习技术，包括**多模态残差网络（Multimodal ResNet）**和**基于特征的对抗网络（DANN）**，以提升目标域数据的分类效果。

## 🛠️ 环境配置需求

### 1. 克隆该项目或下载文件

  ```bash
  git clone https://github.com/DC0827/bearing_diagnosis.git
  ```

### 2. 创建并激活虚拟环境

   ```bash
   conda create -n bearing_diagnosis python=3.8
   conda activate bearing_diagnosis
   ```

### 3. 安装所需依赖

   ```bash
   pip install -r requirements.txt
   ```

## 📋文件说明

以下是项目中的重要文件说明：

### 1. 数据文件

- **DATA_32K**：包含训练集数据采样至32K后的.mat文件。
- **DATA_target**：包含目标域数据的.mat文件。
- **datasets**：存储源域和目标域数据的文件夹。
  - `features_target.csv`: 目标域的特征数据
  - `features.csv`: 源域的特征数据

### 2. 结果文件

- **fig**：包含各类算法的混淆矩阵和可视化图表。

  - `K-Nearest Neighbors_confusion_matrix.png`：KNN的混淆矩阵。
  - `Logistic Regression_confusion_matrix.png`：逻辑回归的混淆矩阵。
  - `my_plot.png`：部分特征的可视化。
  - `Random Forest_confusion_matrix.png`：随机森林的混淆矩阵。
  - `Support Vector Machine_confusion_matrix.png`：SVM的混淆矩阵。
- **model**：包含多模态残差网络模型模型的权重和标准化器。

  - `best_model.pth`：最优模型权重。
  - `scaler.pkl`：标准化器。

### 3. 训练Python 脚本

- `Step1_Segment_target.py`：针对目标域数据进行分割的脚本。
- `Step1_Segment.py`：对源域数据进行分割的脚本。
- `Step2_Grad.py`：计算梯度的脚本。
- `Step2_STFT.py`：针对窗口的短时傅里叶变换提取图片。
- `Step2_Train_ML.py`：机器学习模型的训练与评估。
- `Step2_Train_MultimodalResNet.py`：多模态残差网络的训练与评估。
- `Step2_TSne.py`：进行t-SNE降维可视化多模态残差网络的结果。
- `MultimodalResNet.py`：多模态残差网络的模型架构。
- **step3_transfer_learning**：
  - `correction_details.csv`：二次修正的详细结果。
  - `DANN_visualization.png`：DANN 模型的可视化结果。
  - `detailed_diagnosis_report_with_correction.csv`：修正后的诊断报告。
  - `file_label_counts_with_max_label.csv`：预测域样本的标签计数。
  - `read_result.py`：读取目标域数据集的预测结果。
  - `Step3_transfer_learning.py`：迁移学习的训练。
  - `Step4_transfer_eval.py`：迁移学习评估。
  - `target_domain_predictions.csv`：目标域预测结果文件。
  - `training_loss.png`：训练损失曲线图。
  - `tsne_features_after_transfer.npy`：t-SNE特征。
  - `model_results.txt`：机器学习模型评估结果。

### 4. 其他

- **README.md**：本项目的说明文档。
- **requirements.txt**：项目所需的 Python 包及版本。

## 🌱  数据处理与特征提取

在此步骤中，我们对源域和目标域数据进行了预处理。运行以下脚本处理：

### 1. 源域数据的数据加载与特征提取

```bash
python Step1_Segment.py
```

### 2. 目标域数据的数据加载与特征提取

```bash
python Step1_Segment_target.py
```

## 🤖 机器学习模型训练与测试

本项目使用了四种种传统的机器学习算法（如支持向量机、随机森林等）对轴承故障进行分类。运行以下脚本处理：

```bash
python Step2_Train_ML.py
```

## 🏋️‍♂️ 多模态残差网络训练与模型评估

为了提升分类准确率，本项目使用了**多模态残差网络（Multimodal ResNet）**。运行以下脚本处理：

### 1. STFT 得到图像数据

```bash
python Step2_STFT.py
```

### 2. 模型训练和结果预测

```bash
python Step2_Train_MultimodalResNet.py
```

### 3. 模型结果可视化展示

训练过程卷积特征热力图展示

```bash
python Step2_Grad.py
```

样本嵌入表示的降维可视化展示

```bash
python Step2_Tsen.py
```

## 🔄 基于特征的对抗网络迁移学习训练

在该步骤中，使用了 **基于特征的对抗网络(DANN)** 进行迁移学习，旨在减少源域和目标域之间的分布差异。具体步骤如下：

### 1. DANN模型建立与评估

```bash
python Step3_transfer_learning.py
```

### 2. 目标域样本预测结果读取

```bash
python read_result.py
```

## 📊 迁移学习评估

在迁移学习训练完成后，我们对迁移学习的性能进行评估和可解释分析：

```bash
python Step4_transfer_eval.py
```

## 📩 联系我们

如有任何问题，欢迎通过电子邮件联系我们：
📧 **Email**: [2967228731@qq.com](mailto:2967228731@qq.com)
