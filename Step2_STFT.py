import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, hilbert, stft
import os
from tqdm import tqdm

# --- 设置参数 ---
INPUT_DIR = './DATA_32K'
OUTPUT_DIR = './datasets'
SAMPLING_FREQUENCY = 32000  # 12 kHz
WINDOW_SIZE = 1280          # 0.04 秒 = 12000 * 0.04

# 包络分析参数
BANDPASS_LOW = 1000  # 带通滤波器低频截止频率 (Hz)
BANDPASS_HIGH = 5000 # 带通滤波器高频截止频率 (Hz)

# STFT 参数 (针对包络信号)
# nperseg 通常设置为信号长度，因为我们只分析一个窗口的信号
NFFT_ENVELOPE = WINDOW_SIZE
OVERLAP_ENVELOPE = WINDOW_SIZE // 2

# --- 1. 确保输出文件夹结构存在 ---
def create_output_dirs(labels):
    """为每个类别创建输出文件夹"""
    for label in labels:
        path = os.path.join(OUTPUT_DIR, label)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"创建文件夹: {path}")

# --- 2. 获取文件路径和标签 ---
def get_file_paths_and_labels():
    """动态遍历输入文件夹，获取所有 .mat 文件的路径"""
    file_info = []
    labels = []
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 输入文件夹 '{INPUT_DIR}' 不存在。")
        return file_info, labels

    for label in os.listdir(INPUT_DIR):
        label_dir = os.path.join(INPUT_DIR, label)
        if os.path.isdir(label_dir):
            labels.append(label)
            for filename in os.listdir(label_dir):
                if filename.endswith('.mat'):
                    file_path = os.path.join(label_dir, filename)
                    # 将文件名（不含扩展名）和标签一起返回
                    file_info.append((file_path, os.path.splitext(filename)[0], label))
    return file_info, labels

# --- 3. 读取信号和切片 ---
def process_mat_file(file_path, file_name, label):
    """读取 .mat 文件，切片并生成包络谱图"""
    try:
        matlab_file = scipy.io.loadmat(file_path)
        keys = [k for k in matlab_file if k.endswith('_DE_time')]
        if not keys:
            print(f"警告: 文件 '{file_path}' 中未找到时域信号。")
            return

        signal = matlab_file[keys[0]].reshape(-1)
        num_segments = len(signal) // WINDOW_SIZE

        # 包络分析核心参数
        nyquist = 0.5 * SAMPLING_FREQUENCY
        low_cutoff = BANDPASS_LOW / nyquist
        high_cutoff = BANDPASS_HIGH / nyquist
        b, a = butter(4, [low_cutoff, high_cutoff], btype='band')

        for i in range(num_segments):
            start = i * WINDOW_SIZE
            end = start + WINDOW_SIZE
            segment = signal[start:end]

            # 1. 对切片应用带通滤波器
            filtered_signal = lfilter(b, a, segment)

            # 2. 希尔伯特变换获取包络
            analytic_signal = hilbert(filtered_signal)
            amplitude_envelope = np.abs(analytic_signal)

            # 3. 对包络信号执行 STFT，生成包络时频图
            f, t, Zxx = stft(
                amplitude_envelope,
                fs=SAMPLING_FREQUENCY,
                window='hann',
                nperseg=NFFT_ENVELOPE,
                noverlap=OVERLAP_ENVELOPE
            )
            Zxx_abs = np.abs(Zxx)

            # 保存为图像
            plt.figure(figsize=(2.56, 2.56), dpi=100) # 256x256 像素
            # 这里的t和f是针对STFT的，所以plt.pcolormesh使用它们
            plt.pcolormesh(t, f, np.log(Zxx_abs + 1e-10), shading='auto')
            plt.axis('off') # 隐藏坐标轴
            plt.tight_layout(pad=0) # 移除多余边界

            # 使用 file_index 格式生成图片名
            output_filename = f"{file_name}_{i}.png"
            output_path = os.path.join(OUTPUT_DIR, label, output_filename)

            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

    except Exception as e:
        print(f"处理文件 {file_path} 时发生错误: {e}")

# --- 主执行区块 ---
def main():
    print("--- 开始数据转换 (包络时频图) ---")
    file_info, labels = get_file_paths_and_labels()
    if not file_info:
        return

    create_output_dirs(labels)

    for file_path, file_name, label in tqdm(file_info, desc="处理文件"):
        process_mat_file(file_path, file_name, label)

    print("--- 数据转换完成 ---")
    print(f"所有包络时频图已保存至 '{OUTPUT_DIR}' 文件夹。")

if __name__ == "__main__":
    main()
