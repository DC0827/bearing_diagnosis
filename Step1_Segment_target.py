import scipy.io
import numpy as np
import pandas as pd
import os
from Step1_Segment import extract_features

# 根据 32kHz 采样频率和 0.04秒 的时间窗口来定义切分大小
# 32,000 Hz * 0.04 秒 = 480 个点
WINDOW_SIZE = 1280

# --- 1. 动态获取文件路径和标签 ---
def get_files_and_labels(base_dir):
    """
    遍历指定目录，获取所有 .mat 文件路径并以文件夹名作为标签。
    返回一个包含 (文件路径, 标签) 的列表。
    """
    file_info = []
    # 遍历主目录下的所有子目录
    for label in os.listdir(base_dir):
        label_dir = os.path.join(base_dir, label)
        # 确保是目录
        if os.path.isdir(label_dir):
            # 遍历子目录下的所有文件
            for filename in os.listdir(label_dir):
                if filename.endswith('.mat'):
                    file_path = os.path.join(label_dir, filename)
                    file_info.append((file_path, label))
    return file_info

# --- 2. 从Matlab文件中获取数据 ---
def get_tensors_from_matlab(file_info):
    """
    从给定的文件路径列表中加载并提取时域信号。
    返回一个包含 (信号数据, 文件名, 标签) 的列表。
    """
    signals_with_labels = []
    for file_path, label in file_info:
        try:
            # 加载MATLAB文件
            matlab_file = scipy.io.loadmat(file_path)

            # 假设文件中的第一个变量就是信号数据（没有列名，只有一列数据）
            keys = [key for key in matlab_file if not key.startswith('__')]  # 跳过mat文件中的元数据（__开头的键）
            
            if len(keys) > 0:
                # 获取第一个信号数据
                signal = matlab_file[keys[0]].flatten()  # 取第一个变量的数据并展平为一维数组

                # 提取文件名（去掉路径和扩展名）
                file_name = os.path.basename(file_path).replace('.mat', '')

                # 将信号、文件名和标签一起存储
                signals_with_labels.append((signal, file_name, label))
            else:
                print(f"警告：文件 {file_path} 没有信号数据。")

        except FileNotFoundError:
            print(f"警告：文件 {file_path} 不存在，已跳过。")
        except Exception as e:
            print(f"处理文件 {file_path} 时发生错误：{e}")

    # 返回提取的数据
    return signals_with_labels

# --- 3. 定义信号切分函数 ---
def segment_signal(signal, window_size):
    """
    将长信号切分成不重叠的短信号段。
    """
    num_segments = len(signal) // window_size
    segments = np.array_split(signal[:num_segments * window_size], num_segments)
    return segments


# --- 5. 创建CSV文件 ---
def create_csv_with_features(base_dir = None):
    # 动态获取文件路径和标签
    file_info = get_files_and_labels(base_dir)

    # 加载数据
    signals_with_labels = get_tensors_from_matlab(file_info)

    # 准备存储特征、文件索引和标签的列表
    features_list = []
    file_indices_list = []
    labels_list = []

    # 遍历每个完整信号，并将其切分、提取特征
    for signal, file_name, label in signals_with_labels:
        # 将信号切分成多个 0.04秒（480点）的信号段
        segments = segment_signal(signal, WINDOW_SIZE)

        # 遍历每个信号段，提取特征并添加到列表中
        for i, segment in enumerate(segments):
            features = extract_features(segment,600)
            features_list.append(features)
            # 将文件名和段索引合并为一个字符串
            file_indices_list.append(f"{file_name}_{i}")
            labels_list.append(label)

    # 创建DataFrame
    df = pd.DataFrame(features_list)
    df['file_index'] = file_indices_list
    df['label'] = labels_list

    # 调整列的顺序，将新增的列放在前面
    cols = ['file_index'] + [col for col in df.columns if col not in ['file_index', 'label']] + ['label']
    df = df[cols]

    # 保存为CSV文件
    df.to_csv('./datasets/features_target.csv', index=False)
    print("特征提取完成并已保存为 features.csv 文件。")
    print(f"总共提取了 {len(df)} 条数据。")
    print("\nCSV 文件内容预览：")
    print(df.head())

# 运行主函数
if __name__ == "__main__":
    create_csv_with_features('./DATA_target')