import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def main():
    """
    主函数，用于加载数据、按文件划分数据集、训练多个模型、调优参数并评估性能。
    """
    # 文件路径
    file_path = 'datasets/features.csv'

    # --- 1. 加载数据 ---
    if not os.path.exists(file_path):
        print(f"错误：文件 '{file_path}' 不存在。请检查路径是否正确。")
        return

    df = pd.read_csv(file_path)

    print("数据加载成功，预览：")
    print(df.head())
    print("-" * 40)

    # --- 2. 准备特征和标签，并提取唯一文件ID ---
    # file_index 格式为 "文件名_序号"，我们需要提取文件名作为唯一标识符
    df['file_id'] = df['file_index'].apply(lambda x: x.split('_')[0])

    # 编码标签
    labels = df['label']
    labels_encoded, label_map = pd.factorize(labels)
    label_dict = dict(enumerate(label_map))
    print("标签编码映射：", label_dict)
    print("-" * 40)

    # --- 3. 按文件ID划分数据集 ---
    # 获取所有唯一的 file_id
    unique_files = df['file_id'].unique()

    # 划分训练集和验证集的文件ID
    # stratify 参数根据文件的标签分布进行分层抽样，以保持训练集和验证集的标签比例一致
    file_labels = df.drop_duplicates('file_id').set_index('file_id')['label']
    file_labels_encoded, _ = pd.factorize(file_labels)

    train_files, val_files = train_test_split(
        unique_files,
        test_size=0.2,
        random_state=42,
        stratify=file_labels_encoded
    )

    # 根据文件ID筛选出训练集和验证集的数据
    train_df = df[df['file_id'].isin(train_files)]
    val_df = df[df['file_id'].isin(val_files)]

    # 从分割后的DataFrame中准备特征和标签
    # 过滤掉 'label', 'file_index', 'file_id' 列，只保留特征列
    features_cols = [col for col in df.columns if col not in ['label', 'file_index', 'file_id']]
    X_train = train_df[features_cols]
    y_train = pd.factorize(train_df['label'])[0]  # 重新对训练集标签编码以确保正确性

    X_val = val_df[features_cols]
    y_val = pd.factorize(val_df['label'])[0]      # 重新对验证集标签编码以确保正确性

    scaler = StandardScaler()
    # 对训练集数据进行拟合并转换
    X_train= scaler.fit_transform(X_train)

    # 使用训练集的标准化参数对验证集进行转换
    X_val = scaler.transform(X_val)    

    # 输出送入模型训练的特征列
    print("送入模型训练的特征列为：", features_cols)
    print("-" * 40)

    print(f"训练集包含 {len(train_files)} 个文件，共 {X_train.shape[0]} 条数据")
    print(f"验证集包含 {len(val_files)} 个文件，共 {X_val.shape[0]} 条数据")
    print("-" * 40)

    # --- 4. 定义模型和参数 ---
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=3000),
            'params': {
                'C': [0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs']
            }
        },
        'K-Nearest Neighbors': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 80, 100, 150, 200, 250],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        },
        'Support Vector Machine': {
            'model': SVC(random_state=42),
            'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.1, 1],
                    'kernel': ['rbf', 'linear']
            }
        }
    }

    # 创建一个文件用于保存结果
    with open("model_results.txt", "w") as f:
        # --- 5. 网格搜索并训练模型 ---
        for model_name, model_info in models.items():
            print(f"开始训练 {model_name} 模型...")
            grid_search = GridSearchCV(model_info['model'], model_info['params'], cv=5, n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            print(f"{model_name} 模型训练完成！最优参数：{grid_search.best_params_}")
            f.write(f"{model_name} 最优参数: {grid_search.best_params_}\n")

            # --- 6. 评估模型性能并生成报告 ---
            print(f"开始评估 {model_name} 模型...")
            y_pred = best_model.predict(X_val)

            # 评估需要原始标签名称
            accuracy = accuracy_score(y_val, y_pred)
            print(f"{model_name} 模型在验证集上的准确率为: {accuracy:.4f}")
            f.write(f"{model_name} 模型在验证集上的准确率: {accuracy:.4f}\n")

            print(f"{model_name} 分类报告：")
            report = classification_report(y_val, y_pred, target_names=label_map)
            print(report)
            f.write(f"{model_name} 分类报告：\n{report}\n")

            # 绘制混淆矩阵
            cm = confusion_matrix(y_val, y_pred)
            plt.figure(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map, yticklabels=label_map)
            plt.title(f"{model_name} Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.savefig(f"./fig/{model_name}_confusion_matrix.png")
            plt.close()

            print("-" * 40)

if __name__ == "__main__":
    main()



# python Step4_Train.py

