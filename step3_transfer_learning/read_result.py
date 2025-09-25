import pandas as pd

# 假设你已经将数据保存为CSV文件，路径为 'your_data.csv'
# 读取CSV文件
data = pd.read_csv('./detailed_diagnosis_report_with_correction.csv')


# 提取真实文件名（假设每个文件名的前缀是文件的真实标签）
data['Real_Filename'] = data['Filename'].str.extract(r'([A-Za-z]+)')

# 统计每个真实文件对应的子文件标签数量
file_label_counts = data.groupby(['Real_Filename', 'Predicted_Label']).size().unstack(fill_value=0)

# 添加一列最大标签分类的数量，标记为该文件的标签
file_label_counts['Max_Label'] = file_label_counts.idxmax(axis=1)

# 将统计结果保存为CSV文件
file_label_counts.to_csv('file_label_counts_with_max_label.csv')

print("统计结果已保存为 'file_label_counts_with_max_label.csv'")
