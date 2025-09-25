import scipy.io
import numpy as np
import pandas as pd
import os
from scipy.stats import kurtosis, skew
from scipy.fft import fft
import pywt
from scipy.signal import hilbert

# 根据 12kHz 采样频率和 0.04秒 的时间窗口来定义切分大小
# 12,000 Hz * 0.04 秒 = 480 个点
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
            matlab_file = scipy.io.loadmat(file_path)
            # 假设时域信号变量以 '_DE_time' 结尾
            for position in ['DE']:
                keys = [k for k in matlab_file if k.endswith(position + "_time")]
                if len(keys) > 0:
                    array_key = keys[0]
                    # 将信号数据、文件名和标签一起存储
                    signal = matlab_file[array_key].reshape(1, -1)[0]
                    file_name = os.path.basename(file_path).replace('.mat', '')
                    signals_with_labels.append((signal, file_name, label))
        except FileNotFoundError:
            print(f"警告：文件 {file_path} 不存在，已跳过。")
        except Exception as e:
            print(f"处理文件 {file_path} 时发生错误：{e}")
    return signals_with_labels

# --- 3. 定义信号切分函数 ---
def segment_signal(signal, window_size):
    """
    将长信号切分成不重叠的短信号段。
    """
    num_segments = len(signal) // window_size
    segments = np.array_split(signal[:num_segments * window_size], num_segments)
    return segments

# --- 4. 定义特征提取函数 ---
def extract_features(time_series,rpm=1770):
   
    ################################时域特征分析################################
    # T1: 均值
    X_mean = np.mean(time_series)
    # T2: 均方根
    X_rms = np.sqrt(np.mean(np.square(time_series)))
    # T3: 峰值
    X_max = np.max(np.abs(time_series))
    # T4: 峰-峰值
    X_p_p = np.max(time_series) - np.min(time_series)
    # T5: 峭度
    beta = np.mean(np.power(time_series, 4))
    # T6: 波形指标 (除以绝对值)
    s_f = X_rms / np.abs(X_mean) if X_mean != 0 else 0
    # T7: 峰值指标
    C_f = X_max / X_rms if X_rms != 0 else 0
    # T8: 脉冲指标 (除以绝对值)
    I_f = X_max / np.abs(X_mean) if X_mean != 0 else 0
    # T9: 裕度指标
    C_L = X_max / X_rms if X_rms != 0 else 0
    # T10 峭度指标 (Kurtosis) 计算  
    kurt = beta / (X_rms**4)
    # T11偏度 (Skewness) 计算
    skewness = skew(time_series)
    #T12 裂隙因子:clearance_factor
    x=np.asarray(time_series)
    peak=np.max(np.abs(x))
    clearance_factor=float(peak/((np.mean(np.sqrt(np.abs(x)))**2)+1e-12))
    #T13 "过零率:zero_cross_rate
    zero_cross_rate = float(np.mean(np.diff(np.signbit(time_series)) != 0))
    ################################频域特征分析################################
    # 计算解析信号
    analytical_signal = scipy.signal.hilbert(time_series)
    # 获取包络
    envelope = np.abs(analytical_signal)
    # 计算包络的FFT
    fft_envelope = scipy.fft.fft(envelope)
    fft_magnitude = np.abs(fft_envelope)
    # 定义频率轴
    n_samples = len(envelope)
    freq_axis = scipy.fft.fftfreq(n_samples, 1 / 32000)[:n_samples // 2]
    fft_magnitude = fft_magnitude[:n_samples // 2]
    # 定义搜索窗口（±5 Hz）
    window_hz = 5
    # 提取BPFO及其谐波的幅值
    bpfo_amp = 0
    for harmonic in [1, 2, 3]:  # 基频和前两个谐波
        target_freq = 105 * harmonic
        idx_range = np.where((freq_axis >= target_freq - window_hz) &
                             (freq_axis <= target_freq + window_hz))[0]
        if len(idx_range) > 0:
            bpfo_amp = max(bpfo_amp, np.max(fft_magnitude[idx_range]))
    # 提取BPFI及其谐波的幅值
    bpfi_amp = 0
    for harmonic in [1, 2, 3]:
        target_freq = 165 * harmonic
        idx_range = np.where((freq_axis >= target_freq - window_hz) &
                             (freq_axis <= target_freq + window_hz))[0]
        if len(idx_range) > 0:
            bpfi_amp = max(bpfi_amp, np.max(fft_magnitude[idx_range]))
    # 提取BSF及其谐波的幅值
    bsf_amp = 0
    for harmonic in [1, 2, 3]:
        target_freq = 130 * harmonic
        idx_range = np.where((freq_axis >= target_freq - window_hz) &
                             (freq_axis <= target_freq + window_hz))[0]
        if len(idx_range) > 0:
            bsf_amp = max(bsf_amp, np.max(fft_magnitude[idx_range]))
    # 提取转频幅值
    fr_amp = 0
    target_freq = rpm / 60.0
    idx_range = np.where((freq_axis >= target_freq - window_hz) &
                         (freq_axis <= target_freq + window_hz))[0]
    if len(idx_range) > 0:
        fr_amp = np.max(fft_magnitude[idx_range])
    # 计算频谱质心
    spectral_centroid = np.sum(freq_axis * fft_magnitude) / np.sum(fft_magnitude)
    #谱平坦度
    X=np.fft.rfft((time_series-np.mean(time_series))*np.hanning(len(time_series)))
    ps=np.abs(X)**2+1e-12
    spectral_flatness = float(np.exp(np.mean(np.log(ps))) / np.mean(ps))
    #谱带宽
    x = time_series
    X=np.fft.rfft((x-np.mean(x))*np.hanning(len(x)))
    mag=np.abs(X)
    f=np.fft.rfftfreq(len(x), 1/32000)
    wsum=(mag.sum()+1e-12)
    centroid=float((f*mag).sum()/wsum)
    spec_bandwidth=float(np.sqrt(((f-centroid)**2*mag).sum()/wsum))
    #"wav_hi_energy_ratio:"高频能量占比" "wav_hi_kurtosis":"高频峭度",
    coeffs = pywt.wavedec(time_series,'db4', level=4)
    detail_energy = sum(np.sum(c**2) for c in coeffs[1:2])      # 最高频一层
    total_energy  = sum(np.sum(c**2) for c in coeffs)
    wav_hi_energy_ratio = float(detail_energy/(total_energy+1e-12))
    d = coeffs[1].astype(float)
    wav_hi_kurtosis = float(np.mean(((d-d.mean())/(d.std()+1e-12))**4))


    ################################时频域特征分析################################
    # 小波包分析
    wavelet_features = get_wavelet_packet_feature(time_series)


    ################################序列熵特征################################
    # 两类熵
    amp_H = shannon_entropy_amplitude(time_series)
    spec_H = spectral_entropy(time_series)


    ################################包络特征################################
     # 使用希尔伯特变换获取包络
    analytic_signal = hilbert(time_series)
    envelope = np.abs(analytic_signal)
    
    # 最大值和最小值
    max_value = np.max(envelope)
    min_value = np.min(envelope)
    
    # 均值和标准差
    mean_value = np.mean(envelope)
    std_value = np.std(envelope)
    
    # 峰度和偏度
    kurt_value = kurtosis(envelope)
    skew_value = skew(envelope)
    
    # 频谱分析
    N = len(envelope)
    freq = np.fft.fftfreq(N)
    fft_values = np.abs(fft(envelope))
    
    # 包络的频谱特征
    spectrum_peak = np.max(fft_values)  # 频谱峰值
    spectrum_mean = np.mean(fft_values)  # 频谱均值

    # 
    return {
        #时域特征
        "均值 (T1)": X_mean,
        "均方根 (T2)": X_rms,
        "峰值 (T3)": X_max,
        "峰-峰值 (T4)": X_p_p,
        "峭度 (T5)": beta,
        "波形指标 (T6)": s_f,
        "峰值指标 (T7)": C_f,
        "脉冲指标 (T8)": I_f,
        "裕度指标 (T9)": C_L,
        "峭度指标 (T10)": kurt,
        "偏度 (T11)": skewness,
        "裂隙因子(T12)":clearance_factor,
        "过零率(T13)":zero_cross_rate,
        #频域特征
        'BPFO谐波幅值(f1)': bpfo_amp,
        'BPFI谐波幅值(f2)': bpfi_amp,
        'BSF谐波幅值(f3)': bsf_amp,
        '转频幅值(f4)': fr_amp,
        '频谱质心(f5)': spectral_centroid,
        '谱平坦度(f6)':spectral_flatness,   #谱平坦度
        '谱带宽(f7)': spec_bandwidth,   #谱带宽
        "高频能量占比(f8)":wav_hi_energy_ratio, #  高频能量占比
        "高频峭度(f9)":wav_hi_kurtosis,  #   高频峭度
        #时频域特征
        "小波包特征1": wavelet_features[0],
        "小波包特征2": wavelet_features[1],
        "小波包特征3": wavelet_features[2],
        "小波包特征4": wavelet_features[3],
        "小波包特征5": wavelet_features[4],
        "小波包特征6": wavelet_features[6],
        "小波包特征7": wavelet_features[6],
        "小波包特征8": wavelet_features[7],
        #包络特征
        '包络最大值(e1)': max_value,
        '包络最小值(e2)': min_value,
        '包络均值(e3)': mean_value,
        '包络标准差(e4)': std_value,
        '包络峰度(e5)': kurt_value,
        '包络偏度(e6)': skew_value,
        '包络频谱峰值(e7)': spectrum_peak,
        '包络频谱均值(e8)': spectrum_mean,
        #熵特征
        "幅值熵(s1)": amp_H,   
        "谱熵(s1)": spec_H
    }


# =============== 核心计算 ===============
def shannon_entropy_amplitude(x: np.ndarray, bins: int = 256) -> float:
    x = np.asarray(x).ravel()
    if x.size == 0: return np.nan
    hist, _ = np.histogram(x, bins=bins, density=True)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def spectral_entropy(x: np.ndarray) -> float:
    x = np.asarray(x).ravel()
    if x.size == 0: return np.nan
    w = np.hanning(x.size)
    X = np.fft.rfft((x - x.mean()) * w)
    psd = (np.abs(X) ** 2)
    psd /= (psd.sum() + 1e-12)
    psd = psd[psd > 0]
    return float(-np.sum(psd * np.log2(psd)))

def get_wavelet_packet_feature(data, wavelet='db3', mode='symmetric', maxlevel=3):
    """
    提取 小波包特征
    @param data: shape 为 (n, ) 的 1D array 数据，其中，n 为样本（信号）长度
    @return: 最后一层 子频带 的 能量百分比
    """

    # 执行3级小波包变换
    wavelet = 'db1'
    max_level = 3
    wp = pywt.WaveletPacket(data, wavelet=wavelet, mode='symmetric', maxlevel=max_level)

    # 获取叶节点（第3级共有2^3=8个节点）
    nodes = [node.path for node in wp.get_level(max_level, 'natural')]

    # 计算每个节点的能量
    energies = []
    for i, node_path in enumerate(nodes, 1):
        node = wp[node_path]
        coeffs = node.data
        energy = np.sum(coeffs ** 2)
        energies.append(energy)
    return np.array(energies)



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
            features = extract_features(segment)
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
    df.to_csv('./datasets/features.csv', index=False)
    print("特征提取完成并已保存为 features.csv 文件。")
    print(f"总共提取了 {len(df)} 条数据。")
    print("\nCSV 文件内容预览：")
    print(df.head())

# 运行主函数
if __name__ == "__main__":
    create_csv_with_features('./DATA_32K')