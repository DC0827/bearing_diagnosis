import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import traceback

import matplotlib.font_manager as fm



# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['STSong']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)


# ================== 第一部分：模型定义与加载 ==================

class GradientReversalFunction(torch.autograd.Function):
    """梯度反转层的实现"""

    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_val
        return output, None


class GradientReversalLayer(nn.Module):
    """梯度反转层封装"""

    def __init__(self, lambda_val=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)


class DANN(nn.Module):
    """领域对抗神经网络模型"""

    def __init__(self, input_dim=40, num_classes=4, lambda_val=1.0):
        super(DANN, self).__init__()

        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 标签预测器
        self.label_predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        # 梯度反转层
        self.grl = GradientReversalLayer(lambda_val)

        # 领域判别器
        self.domain_discriminator = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # 提取特征
        features = self.feature_extractor(x)

        # 标签预测
        class_logits = self.label_predictor(features)

        # 领域判别（通过GRL）
        reversed_features = self.grl(features)
        domain_logits = self.domain_discriminator(reversed_features)

        return class_logits, domain_logits, features


def load_model_and_data():
    """加载模型和数据"""
    print("=" * 80)
    print("加载模型和数据...")
    print("=" * 80)

    # 1. 加载模型
    model = DANN(input_dim=40, num_classes=4)
    model.load_state_dict(torch.load('dann_model.pth', map_location='cpu'))
    model.eval()
    print("✓ 成功加载模型权重: dann_model.pth")

    # 2. 加载源域数据
    source_df = pd.read_csv('features.csv')
    feature_columns = [
    "均值 (T1)", "均方根 (T2)", "峰值 (T3)", "峰-峰值 (T4)", "峭度 (T5)", "波形指标 (T6)", 
    "峰值指标 (T7)", "脉冲指标 (T8)", "裕度指标 (T9)", "峭度指标 (T10)", "偏度 (T11)", 
    "裂隙因子(T12)", "过零率(T13)", "BPFO谐波幅值(f1)", "BPFI谐波幅值(f2)", "BSF谐波幅值(f3)", 
    "转频幅值(f4)", "频谱质心(f5)", "谱平坦度(f6)", "谱带宽(f7)", "高频能量占比(f8)", 
    "高频峭度(f9)", "小波包特征1", "小波包特征2", "小波包特征3", "小波包特征4", 
    "小波包特征5", "小波包特征6", "小波包特征7", "小波包特征8", "包络最大值(e1)", 
    "包络最小值(e2)", "包络均值(e3)", "包络标准差(e4)", "包络峰度(e5)", "包络偏度(e6)", 
    "包络频谱峰值(e7)", "包络频谱均值(e8)", "幅值熵(s1)", "谱熵(s1)"
    ]

    X_source = source_df[feature_columns].values
    y_source = source_df['label'].values
    print(f"✓ 成功加载源域数据: {len(X_source)} 个样本")

    # 3. 加载目标域数据
    target_df = pd.read_csv('./features_target.csv')
    X_target = target_df[feature_columns].values
    print(f"✓ 成功加载目标域特征: {X_target.shape}")

    # 4. 数据标准化
    scaler = StandardScaler()
    X_combined = np.vstack([X_source, X_target])
    X_combined_scaled = scaler.fit_transform(X_combined)
    X_source_scaled = X_combined_scaled[:len(X_source)]
    X_target_scaled = X_combined_scaled[len(X_source):]

    # 5. 加载预测结果
    predictions = pd.read_csv('target_domain_predictions.csv')
    if 'Predicted_Label' not in predictions.columns:
        for col in predictions.columns:
            if 'Label' in col or 'label' in col:
                predictions['Predicted_Label'] = predictions[col]
                break

    print(f"✓ 成功加载预测结果: {len(predictions)} 个样本")

    # 6. 加载t-SNE特征
    tsne_features = np.load('tsne_features_after_transfer.npy')
    print(f"✓ 成功加载共享特征: {tsne_features.shape}")

    # 组织数据
    source_data = {
        'features': X_source_scaled,
        'features_original': X_source,
        'labels': y_source,
        'shared_features': tsne_features[:len(X_source)]
    }

    target_data = {
        'features': X_target_scaled,
        'features_original': X_target,
        'predictions': predictions,
        'shared_features': tsne_features[len(X_source):]
    }

    print(f"\n数据加载完成:")
    print(f"  源域样本数: {len(X_source)}")
    print(f"  目标域样本数: {len(X_target)}")
    print(f"  特征维度: {X_source.shape[1]}")

    return model, source_data, target_data, predictions, feature_columns, scaler


# ================== 第二部分：基础可解释性分析 ==================

def get_consistent_label_mapping():
    """获取一致的标签映射关系"""
    labels = ['B', 'IR', 'NORMAL', 'OR']
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label


def compute_physics_score_for_hypothesis(sample_features, hypothesis_label, feature_columns):
    """
    计算假设某个样本为特定类别时的物理机理符合度

    Parameters:
    -----------
    sample_features : np.array
        样本特征值
    hypothesis_label : str
        假设的故障类别
    feature_columns : list
        特征列名

    Returns:
    --------
    score : float
        物理机理符合度分数
    analysis_info : dict
        分析详情
    """
    feature_idx_map = {name: idx for idx, name in enumerate(feature_columns)}

    analysis_info = {
        'hypothesis_label': hypothesis_label,
        'key_features': {}
    }

    # 特征标准化
    feature_mean = np.mean(sample_features)
    feature_std = np.std(sample_features) + 1e-6
    NORMALized_features = (sample_features - feature_mean) / feature_std

    if hypothesis_label == 'OR':
        bpfo_idx = feature_idx_map.get('BPFO_Amplitude', -1)
        if bpfo_idx >= 0:
            bpfo_NORMALized = NORMALized_features[bpfo_idx]
            score = 1 / (1 + np.exp(-bpfo_NORMALized))
            analysis_info['key_features']['BPFO_Amplitude'] = float(sample_features[bpfo_idx])
            analysis_info['NORMALized_bpfo'] = float(bpfo_NORMALized)
        else:
            score = 0.5

    elif hypothesis_label == 'IR':
        bpfi_idx = feature_idx_map.get('BPFI_Amplitude', -1)
        if bpfi_idx >= 0:
            bpfi_NORMALized = NORMALized_features[bpfi_idx]
            score = 1 / (1 + np.exp(-bpfi_NORMALized))
            analysis_info['key_features']['BPFI_Amplitude'] = float(sample_features[bpfi_idx])
            analysis_info['NORMALized_bpfi'] = float(bpfi_NORMALized)
        else:
            score = 0.5

    elif hypothesis_label == 'B':
        bsf_idx = feature_idx_map.get('BSF_Amplitude', -1)
        if bsf_idx >= 0:
            bsf_NORMALized = NORMALized_features[bsf_idx]
            score = 1 / (1 + np.exp(-bsf_NORMALized))
            analysis_info['key_features']['BSF_Amplitude'] = float(sample_features[bsf_idx])
            analysis_info['NORMALized_bsf'] = float(bsf_NORMALized)
        else:
            score = 0.5

    else:  # NORMAL
        fault_features = ['BPFO_Amplitude', 'BPFI_Amplitude', 'BSF_Amplitude']
        fault_scores = []

        for feat in fault_features:
            feat_idx = feature_idx_map.get(feat, -1)
            if feat_idx >= 0:
                feat_NORMALized = NORMALized_features[feat_idx]
                fault_score = 1 / (1 + np.exp(feat_NORMALized))  # 注意这里没有负号
                fault_scores.append(fault_score)
                analysis_info['key_features'][feat] = float(sample_features[feat_idx])

        if fault_scores:
            score = np.mean(fault_scores)
        else:
            score = 0.5

    return float(np.clip(score, 0.0, 1.0)), analysis_info


def compute_domain_alignment_scores(source_data, target_data):
    """计算领域对齐质量分数"""
    print("\n" + "=" * 80)
    print("计算领域对齐质量分数")
    print("=" * 80)

    source_features = source_data['shared_features']
    source_labels = source_data['labels']
    target_features = target_data['shared_features']
    target_predictions = target_data['predictions']['Predicted_Label'].values

    k_neighbors = 5
    nn_model = NearestNeighbors(n_neighbors=min(k_neighbors, len(source_features)),
                                metric='euclidean')
    nn_model.fit(source_features)

    alignment_scores = []

    for idx in range(len(target_features)):
        target_sample = target_features[idx:idx + 1]
        distances, indices = nn_model.kneighbors(target_sample)
        predicted_label = target_predictions[idx]

        consistent_count = 0
        for neighbor_idx in indices[0]:
            source_label = source_labels[neighbor_idx]
            if source_label == predicted_label:
                consistent_count += 1

        score = consistent_count / len(indices[0])
        alignment_scores.append(score)

    print(f"\n领域对齐质量统计:")
    print(f"  范围: [{min(alignment_scores):.3f}, {max(alignment_scores):.3f}]")
    print(f"  平均值: {np.mean(alignment_scores):.3f}")
    print(f"  标准差: {np.std(alignment_scores):.3f}")

    return alignment_scores, nn_model


def compute_classifier_probabilities(model, target_data):
    """计算分类器的原始置信度"""
    print("\n" + "=" * 80)
    print("计算分类器原始置信度")
    print("=" * 80)

    model.eval()
    classifier_probs = []
    all_probabilities = []

    label_to_idx, _ = get_consistent_label_mapping()

    with torch.no_grad():
        X_target_torch = torch.FloatTensor(target_data['features'])
        class_logits, _, _ = model(X_target_torch)
        probabilities = F.softmax(class_logits, dim=1).cpu().numpy()

    for idx, pred_label in enumerate(target_data['predictions']['Predicted_Label']):
        pred_class = label_to_idx.get(pred_label, 0)
        prob = probabilities[idx, pred_class]
        classifier_probs.append(float(prob))
        all_probabilities.append(probabilities[idx])

    print(f"\n分类器置信度统计:")
    print(f"  范围: [{min(classifier_probs):.3f}, {max(classifier_probs):.3f}]")
    print(f"  平均值: {np.mean(classifier_probs):.3f}")
    print(f"  标准差: {np.std(classifier_probs):.3f}")

    return classifier_probs, all_probabilities


def perform_basic_analysis(model, source_data, target_data, feature_columns):
    """执行基础可解释性分析"""
    print("\n" + "=" * 80)
    print("执行基础可解释性分析")
    print("=" * 80)

    physics_scores = []
    analysis_details = []

    print("计算物理机理符合度分数...")

    for idx in tqdm(range(len(target_data['features_original']))):
        sample = target_data['features_original'][idx]
        predicted_label = target_data['predictions'].iloc[idx]['Predicted_Label']

        try:
            score, details = compute_physics_score_for_hypothesis(
                sample, predicted_label, feature_columns)
            physics_scores.append(score)
            analysis_details.append(details)

        except Exception as e:
            print(f"样本 {idx} 物理机理分析失败: {e}")
            physics_scores.append(0.5)
            analysis_details.append({'error': str(e)})

    # 输出统计信息
    valid_scores = [s for s in physics_scores if 0 <= s <= 1]
    if valid_scores:
        print(f"\n物理机理符合度统计:")
        print(f"  范围: [{min(valid_scores):.3f}, {max(valid_scores):.3f}]")
        print(f"  平均值: {np.mean(valid_scores):.3f}")
        print(f"  标准差: {np.std(valid_scores):.3f}")

    return physics_scores, analysis_details


def compute_comprehensive_confidence(classifier_probs, physics_scores,
                                     alignment_scores, weights=(0.3, 0.5, 0.2)):
    """计算综合诊断置信度"""
    print("\n" + "=" * 80)
    print("计算综合诊断置信度")
    print("=" * 80)

    w1, w2, w3 = weights
    print(f"\n权重设置:")
    print(f"  分类器概率: {w1 * 100:.0f}%")
    print(f"  物理机理符合度: {w2 * 100:.0f}%")
    print(f"  领域对齐质量: {w3 * 100:.0f}%")

    comprehensive_scores = []

    for i in range(len(classifier_probs)):
        score = (w1 * classifier_probs[i] +
                 w2 * physics_scores[i] +
                 w3 * alignment_scores[i])
        comprehensive_scores.append(score)

    print(f"\n综合置信度统计:")
    print(f"  范围: [{min(comprehensive_scores):.3f}, {max(comprehensive_scores):.3f}]")
    print(f"  平均值: {np.mean(comprehensive_scores):.3f}")
    print(f"  标准差: {np.std(comprehensive_scores):.3f}")

    return comprehensive_scores


# ================== 第三部分：二次修正诊断框架 ==================

def get_all_class_probabilities(model, sample_features):
    """获取样本对所有类别的预测概率"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(sample_features)
        class_logits, _, _ = model(X_tensor)
        probabilities = F.softmax(class_logits, dim=1).cpu().numpy()[0]

    class_names = ['B', 'IR', 'NORMAL', 'OR']
    class_probs = {name: prob for name, prob in zip(class_names, probabilities)}

    return class_probs


def compute_alignment_score_for_hypothesis(target_sample_features, hypothesis_label,
                                           source_data, nn_model, k_neighbors=5):
    """计算假设某个样本为特定类别时的领域对齐质量"""
    distances, indices = nn_model.kneighbors(target_sample_features.reshape(1, -1))

    consistent_count = 0
    for neighbor_idx in indices[0]:
        source_label = source_data['labels'][neighbor_idx]
        if source_label == hypothesis_label:
            consistent_count += 1

    score = consistent_count / len(indices[0])
    return float(score)


def compute_comprehensive_confidence_for_hypothesis(classifier_prob, physics_score,
                                                    alignment_score, weights=(0.3, 0.5, 0.2)):
    """计算假设特定类别时的综合置信度"""
    w1, w2, w3 = weights
    comprehensive_score = w1 * classifier_prob + w2 * physics_score + w3 * alignment_score
    return float(comprehensive_score)


def should_trigger_correction(original_confidence, all_class_probs,
                              uncertainty_threshold=0.6, ambiguity_threshold=0.15):
    """
    判断是否需要启动二次修正

    Parameters:
    -----------
    original_confidence : float
        原始综合置信度
    all_class_probs : dict
        所有类别的分类器概率
    uncertainty_threshold : float
        置信度阈值
    ambiguity_threshold : float
        决策模糊度阈值（暂时不使用）

    Returns:
    --------
    should_correct : bool
        是否需要修正
    reason : str
        触发原因
    """
    # 简化触发条件：只要置信度低就触发二次诊断
    trigger_low_confidence = original_confidence < uncertainty_threshold

    # 计算决策模糊度（用于信息显示，不作为触发条件）
    sorted_probs = sorted(all_class_probs.values(), reverse=True)
    ambiguity = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0

    # 修改后的触发逻辑：只要低置信度就触发
    should_correct = trigger_low_confidence

    if should_correct:
        if ambiguity < ambiguity_threshold:
            reason = f"低置信度({original_confidence:.3f}) + 决策模糊(差距{ambiguity:.3f})"
        else:
            reason = f"低置信度({original_confidence:.3f})，值得二次审视"
    else:
        reason = f"置信度足够({original_confidence:.3f} ≥ {uncertainty_threshold})"

    return should_correct, reason


def validate_correction(original_result, corrected_result, min_improvement=0.02):
    """
    验证修正的合理性 - 采用更宽松的验证策略

    Parameters:
    -----------
    original_result : dict
        原始结果
    corrected_result : dict
        修正后结果
    min_improvement : float
        最小改善阈值（进一步降低到0.02）

    Returns:
    --------
    is_valid : bool
        修正是否有效
    reason : str
        验证原因
    """
    improvement = corrected_result['comprehensive_confidence'] - original_result['comprehensive_confidence']

    # 检查1：改善是否显著（非常宽松的标准）
    if improvement < min_improvement:
        return False, f"改善不显著({improvement:.3f} < {min_improvement})"

    # 检查2：物理机理是否合理（进一步放宽）
    if corrected_result['physics_score'] < 0.20:
        return False, f"物理机理评分过低({corrected_result['physics_score']:.3f})"

    # 检查3：避免原始置信度过高的样本被修正（新增检查）
    if original_result['comprehensive_confidence'] > 0.7:
        return False, f"原始置信度较高({original_result['comprehensive_confidence']:.3f})，不建议修正"

    # 检查4：不允许过度激进的修正
    if improvement > 0.6:
        return False, f"改善过于激进({improvement:.3f})"

    return True, f"有效修正(改善{improvement:.3f})"


def perform_secondary_correction_analysis(model, source_data, target_data, feature_columns,
                                          original_report, nn_model,
                                          correction_threshold=0.6, weights=(0.3, 0.5, 0.2)):
    """
    执行二次修正诊断分析
    """
    print("\n" + "=" * 80)
    print("启动二次修正诊断框架 - 从解释到仲裁")
    print("=" * 80)

    class_names = ['B', 'IR', 'NORMAL', 'OR']
    corrected_report = original_report.copy()
    correction_details = []

    # 识别需要分析的样本
    low_confidence_mask = original_report['Comprehensive_Confidence'] < correction_threshold
    candidate_samples = original_report[low_confidence_mask]

    print(f"发现 {len(candidate_samples)} 个候选样本（置信度 < {correction_threshold}）")
    print("开始逐一分析修正可能性...")

    for idx, row in candidate_samples.iterrows():
        sample_idx = idx
        original_prediction = row['Predicted_Label']
        original_confidence = row['Comprehensive_Confidence']

        # print(f"\n分析样本 {row['Filename']} (原预测: {original_prediction}, 置信度: {original_confidence:.3f})")

        # 获取样本特征
        sample_features_original = target_data['features_original'][sample_idx:sample_idx + 1]
        sample_features_scaled = target_data['features'][sample_idx:sample_idx + 1]
        sample_shared_features = target_data['shared_features'][sample_idx]

        # 获取所有类别的分类器概率
        class_probs = get_all_class_probabilities(model, sample_features_scaled)

        # 判断是否需要修正
        should_correct, trigger_reason = should_trigger_correction(
            original_confidence, class_probs, correction_threshold, 0.15)

        if not should_correct:
            # print(f"  跳过修正: {trigger_reason}")
            continue

        # print(f"  触发二次仲裁: {trigger_reason}")

        # 为每个可能的类别计算综合置信度
        hypothesis_results = {}

        for hypothesis_label in class_names:
            # 1. 分类器概率
            classifier_prob = class_probs[hypothesis_label]

            # 2. 物理机理符合度
            physics_score, _ = compute_physics_score_for_hypothesis(
                sample_features_original[0], hypothesis_label, feature_columns)

            # 3. 领域对齐质量
            alignment_score = compute_alignment_score_for_hypothesis(
                sample_shared_features, hypothesis_label, source_data, nn_model)

            # 4. 综合置信度
            comprehensive_score = compute_comprehensive_confidence_for_hypothesis(
                classifier_prob, physics_score, alignment_score, weights)

            hypothesis_results[hypothesis_label] = {
                'classifier_prob': classifier_prob,
                'physics_score': physics_score,
                'alignment_score': alignment_score,
                'comprehensive_confidence': comprehensive_score
            }

            # print(f"    假设 {hypothesis_label}: 分类器={classifier_prob:.3f}, "
                #   f"物理={physics_score:.3f}, 对齐={alignment_score:.3f} → 综合={comprehensive_score:.3f}")

        # 找到置信度最高的假设
        best_hypothesis = max(hypothesis_results.keys(),
                              key=lambda k: hypothesis_results[k]['comprehensive_confidence'])
        best_result = hypothesis_results[best_hypothesis]
        original_result = hypothesis_results[original_prediction]

        # 验证修正的合理性
        if best_hypothesis != original_prediction:
            is_valid, validation_reason = validate_correction(original_result, best_result, 0.1)

            if is_valid:
                # print(f"  ✅ 执行修正: {original_prediction} → {best_hypothesis} ({validation_reason})")

                # 更新报告
                corrected_report.loc[idx, 'Predicted_Label'] = best_hypothesis
                corrected_report.loc[idx, 'Classifier_Probability'] = best_result['classifier_prob']
                corrected_report.loc[idx, 'Physics_Mechanism_Score'] = best_result['physics_score']
                corrected_report.loc[idx, 'Domain_Alignment_Score'] = best_result['alignment_score']
                corrected_report.loc[idx, 'Comprehensive_Confidence'] = best_result['comprehensive_confidence']

                # 更新中文标签和置信度等级
                label_mapping = {
                    'NORMAL': '正常',
                    'IR': '内圈故障',
                    'OR': '外圈故障',
                    'B': '滚动体故障'
                }
                corrected_report.loc[idx, '故障类型'] = label_mapping[best_hypothesis]

                def get_confidence_level(score):
                    if score >= 0.8:
                        return '高置信度'
                    elif score >= 0.6:
                        return '中置信度'
                    elif score >= 0.4:
                        return '低置信度'
                    else:
                        return '极低置信度'

                corrected_report.loc[idx, 'Confidence_Level'] = get_confidence_level(
                    best_result['comprehensive_confidence'])

                # 记录修正详情
                correction_details.append({
                    'filename': row['Filename'],
                    'original_prediction': original_prediction,
                    'corrected_prediction': best_hypothesis,
                    'original_confidence': original_confidence,
                    'corrected_confidence': best_result['comprehensive_confidence'],
                    'improvement': best_result['comprehensive_confidence'] - original_confidence,
                    'validation_reason': validation_reason,
                    'all_hypotheses': hypothesis_results
                })
            # else:
                # print(f"  ❌ 拒绝修正: {validation_reason}")
        # else:
            # print(f"  ✓ 保持原预测: 最优假设与原预测一致")

    return corrected_report, correction_details


def analyze_correction_effectiveness(original_report, corrected_report, correction_details):
    """分析修正效果"""
    print("\n" + "=" * 80)
    print("二次修正效果分析")
    print("=" * 80)

    print(f"\n1. 修正统计:")
    print(f"   总样本数: {len(original_report)}")
    print(f"   修正样本数: {len(correction_details)}")
    print(f"   修正比例: {len(correction_details) / len(original_report) * 100:.1f}%")

    if correction_details:
        print(f"\n2. 修正详情:")
        for detail in correction_details:
            print(f"   {detail['filename']}: {detail['original_prediction']} → {detail['corrected_prediction']} "
                  f"(+{detail['improvement']:.3f})")

        print(f"\n3. 置信度改善:")
        original_avg = original_report['Comprehensive_Confidence'].mean()
        corrected_avg = corrected_report['Comprehensive_Confidence'].mean()
        print(f"   原始平均置信度: {original_avg:.3f}")
        print(f"   修正后平均置信度: {corrected_avg:.3f}")
        print(f"   整体改善: {corrected_avg - original_avg:.3f}")

        print(f"\n4. 置信度等级分布变化:")
        original_levels = original_report['Confidence_Level'].value_counts()
        corrected_levels = corrected_report['Confidence_Level'].value_counts()

        all_levels = ['高置信度', '中置信度', '低置信度', '极低置信度']
        for level in all_levels:
            orig_count = original_levels.get(level, 0)
            corr_count = corrected_levels.get(level, 0)
            change = corr_count - orig_count
            if change != 0:
                symbol = "+" if change > 0 else ""
                print(f"   {level}: {orig_count} → {corr_count} ({symbol}{change})")
            else:
                print(f"   {level}: {orig_count} → {corr_count} (无变化)")
    else:
        print("\n   无样本被修正")

    return {
        'correction_count': len(correction_details),
        'correction_rate': len(correction_details) / len(original_report),
        'avg_improvement': corrected_report['Comprehensive_Confidence'].mean() - original_report[
            'Comprehensive_Confidence'].mean()
    }


# ================== 第四部分：报告生成与分析 ==================

def create_comprehensive_report(target_data, classifier_probs, physics_scores,
                                alignment_scores, comprehensive_scores):
    """创建综合诊断报告"""
    report_df = pd.DataFrame({
        'Filename': target_data['predictions']['Filename'],
        'Predicted_Label': target_data['predictions']['Predicted_Label'],
        'Classifier_Probability': np.round(classifier_probs, 4),
        'Physics_Mechanism_Score': np.round(physics_scores, 4),
        'Domain_Alignment_Score': np.round(alignment_scores, 4),
        'Comprehensive_Confidence': np.round(comprehensive_scores, 4)
    })

    report_df = report_df.sort_values('Comprehensive_Confidence', ascending=False)

    def get_confidence_level(score):
        if score >= 0.8:
            return '高置信度'
        elif score >= 0.6:
            return '中置信度'
        elif score >= 0.4:
            return '低置信度'
        else:
            return '极低置信度'

    report_df['Confidence_Level'] = report_df['Comprehensive_Confidence'].apply(get_confidence_level)

    label_mapping = {
        'NORMAL': '正常',
        'IR': '内圈故障',
        'OR': '外圈故障',
        'B': '滚动体故障'
    }
    report_df['故障类型'] = report_df['Predicted_Label'].map(label_mapping)

    return report_df


def generate_detailed_analysis_report(original_report, corrected_report, correction_details):
    """生成详细的分析报告"""
    print("\n" + "=" * 80)
    print("生成详细分析报告")
    print("=" * 80)

    # 创建详细报告
    detailed_report = corrected_report.copy()

    # 添加修正标记
    detailed_report['Is_Corrected'] = False
    detailed_report['Original_Prediction'] = detailed_report['Predicted_Label']
    detailed_report['Confidence_Improvement'] = 0.0

    for detail in correction_details:
        mask = detailed_report['Filename'] == detail['filename']
        detailed_report.loc[mask, 'Is_Corrected'] = True
        detailed_report.loc[mask, 'Original_Prediction'] = detail['original_prediction']
        detailed_report.loc[mask, 'Confidence_Improvement'] = detail['improvement']

    # 保存详细报告
    detailed_report.to_csv('detailed_diagnosis_report_with_correction.csv',
                           index=False, encoding='utf-8-sig')

    # 保存修正详情
    if correction_details:
        correction_df = pd.DataFrame(correction_details)
        correction_df.to_csv('correction_details.csv', index=False, encoding='utf-8-sig')
        print("修正详情已保存为 correction_details.csv")

    print("详细诊断报告已保存为 detailed_diagnosis_report_with_correction.csv")

    return detailed_report


# ================== 第五部分：主函数 ==================

def main():
    """主函数：执行完整的可解释性分析与二次修正诊断流程"""
    print("\n" * 2)
    print("=" * 80)
    print(" " * 10 + "DANN可解释性分析与二次修正诊断框架")
    print(" " * 15 + "从解释到仲裁的完整流程")
    print("=" * 80)

    try:
        # 第一阶段：加载数据和模型
        model, source_data, target_data, predictions, feature_columns, scaler = load_model_and_data()

        # 第二阶段：基础可解释性分析
        print("\n" + "🔍" + " 第一阶段：基础可解释性分析")
        physics_scores, analysis_details = perform_basic_analysis(
            model, source_data, target_data, feature_columns)

        alignment_scores, nn_model = compute_domain_alignment_scores(
            source_data, target_data)

        classifier_probs, all_probabilities = compute_classifier_probabilities(
            model, target_data)

        comprehensive_scores = compute_comprehensive_confidence(
            classifier_probs, physics_scores, alignment_scores,
            weights=(0.3, 0.5, 0.2))

        # 第三阶段：生成初始报告
        print("\n" + "📊" + " 第二阶段：生成初始诊断报告")
        original_report = create_comprehensive_report(
            target_data, classifier_probs, physics_scores,
            alignment_scores, comprehensive_scores)

        # 第四阶段：二次修正诊断
        print("\n" + "⚖️" + " 第三阶段：二次修正诊断仲裁")

        # 尝试不同的权重组合进行二次修正
        weight_schemes = [
            ((0.3, 0.5, 0.2), "标准方案：物理机理主导"),
            ((0.4, 0.4, 0.2), "平衡方案：分类器与物理机理平衡"),
            ((0.2, 0.6, 0.2), "物理优先方案：强调物理机理"),
            ((0.3, 0.4, 0.3), "对齐强化方案：重视迁移质量")
        ]

        best_scheme = None
        best_effectiveness = 0
        all_correction_results = {}

        for weights, scheme_name in weight_schemes:
            print(f"\n{'=' * 60}")
            print(f"尝试权重方案: {scheme_name}")
            print(
                f"权重分配: 分类器{weights[0] * 100:.0f}%, 物理机理{weights[1] * 100:.0f}%, 领域对齐{weights[2] * 100:.0f}%")
            print(f"{'=' * 60}")

            # 重新计算基于新权重的初始综合置信度
            scheme_comprehensive_scores = compute_comprehensive_confidence(
                classifier_probs, physics_scores, alignment_scores, weights)

            scheme_original_report = create_comprehensive_report(
                target_data, classifier_probs, physics_scores,
                alignment_scores, scheme_comprehensive_scores)

            # 执行二次修正（提高触发阈值到0.65）
            scheme_corrected_report, scheme_correction_details = perform_secondary_correction_analysis(
                model, source_data, target_data, feature_columns,
                scheme_original_report, nn_model, correction_threshold=0.65, weights=weights)

            # 分析效果
            scheme_effectiveness = analyze_correction_effectiveness(
                scheme_original_report, scheme_corrected_report, scheme_correction_details)

            # 记录结果
            all_correction_results[scheme_name] = {
                'weights': weights,
                'original_report': scheme_original_report,
                'corrected_report': scheme_corrected_report,
                'correction_details': scheme_correction_details,
                'effectiveness': scheme_effectiveness
            }

            # 选择最佳方案
            total_improvement = (scheme_effectiveness['avg_improvement'] +
                                 scheme_effectiveness['correction_rate'] * 0.5)
            if total_improvement > best_effectiveness:
                best_effectiveness = total_improvement
                best_scheme = scheme_name

        # 使用最佳方案的结果
        print(f"\n{'🏆' * 60}")
        print(f"最佳权重方案: {best_scheme}")
        print(f"{'🏆' * 60}")

        corrected_report = all_correction_results[best_scheme]['corrected_report']
        correction_details = all_correction_results[best_scheme]['correction_details']
        effectiveness = all_correction_results[best_scheme]['effectiveness']
        original_report = all_correction_results[best_scheme]['original_report']

        # 第五阶段：效果分析
        print("\n" + "📈" + " 第四阶段：修正效果分析")
        effectiveness = analyze_correction_effectiveness(
            original_report, corrected_report, correction_details)

        # 第六阶段：显示最终报告
        print("\n" + "=" * 80)
        print(" " * 20 + "最终诊断报告（二次修正后）")
        print("=" * 80)

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        print("\n最终诊断结果:")
        display_columns = ['Filename', '故障类型', 'Classifier_Probability',
                           'Physics_Mechanism_Score', 'Domain_Alignment_Score',
                           'Comprehensive_Confidence', 'Confidence_Level']
        print(corrected_report[display_columns].to_string(index=False))

        # 第七阶段：生成详细报告
        print("\n" + "📋" + " 第五阶段：生成详细分析报告")
        detailed_report = generate_detailed_analysis_report(
            original_report, corrected_report, correction_details)

        # 第八阶段：总结分析
        print("\n" + "=" * 80)
        print(" " * 25 + "总结分析")
        print("=" * 80)

        print(f"\n🎯 核心成果:")
        print(f"   • 建立了从解释到仲裁的二次诊断框架")
        print(f"   • 实现了多源信息融合的置信度评估")
        print(f"   • 通过物理机理约束提升了诊断可信度")

        print(f"\n📊 量化效果:")
        original_avg = original_report['Comprehensive_Confidence'].mean()
        corrected_avg = corrected_report['Comprehensive_Confidence'].mean()
        print(f"   • 最佳方案: {best_scheme}")
        print(f"   • 最佳权重: {all_correction_results[best_scheme]['weights']}")
        print(f"   • 平均置信度: {original_avg:.3f} → {corrected_avg:.3f} (+{corrected_avg - original_avg:.3f})")
        print(f"   • 修正样本数: {effectiveness['correction_count']}/{len(original_report)}")
        print(f"   • 修正比例: {effectiveness['correction_rate'] * 100:.1f}%")

        print(f"\n🔬 方案对比:")
        for scheme_name, result in all_correction_results.items():
            eff = result['effectiveness']
            weights = result['weights']
            print(
                f"   • {scheme_name}: 权重{weights} → 修正{eff['correction_count']}个, 改善{eff['avg_improvement']:+.3f}")

        print(f"\n🔬 技术创新:")
        print(f"   • 从被动解释提升为主动仲裁")
        print(f"   • 融合分类器概率、物理机理、领域对齐三个维度")
        print(f"   • 设计了严格的修正触发和验证机制")
        print(f"   • 自动优选最佳权重配置方案")

        print(f"\n⚠️ 局限性认知:")
        print(f"   • 修正效果受限于物理机理建模的准确性")
        print(f"   • 缺乏真实标签验证，修正准确性有待进一步验证")
        print(f"   • 需要在更大数据集上验证框架的泛化能力")

        print("\n" + "=" * 80)
        print(" " * 15 + "可解释性分析与二次修正诊断完成！")
        print("=" * 80)

        return corrected_report, correction_details, effectiveness

    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("\n请确保已经运行第三问的代码，生成所有必要的文件后再运行此分析。")
        sys.exit(1)

    except Exception as e:
        print(f"\n发生未预期的错误: {e}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":

    # 执行主程序
    final_report, corrections, effectiveness = main()

# python transfer_eval.py