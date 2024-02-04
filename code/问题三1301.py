import pandas as pd
import glob
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 数据准备
folder_path = 'data_with_momentum'
file_list = glob.glob(f'{folder_path}/2023-wimbledon-*.csv')
dataframes = [pd.read_csv(file) for file in file_list]

# 合并所有DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# 从合并后的DataFrame中提取特征
feature_columns = ['p1_sets', 'p2_sets', 'p1_games', 'p2_games', 'server', 'p1_ace', 'p2_ace', 'p1_double_fault', 'p2_double_fault', 'p1_unf_err', 'p2_unf_err', 'p1_break_pt_won', 'p2_break_pt_won', 'p1_break_pt_missed', 'p2_break_pt_missed', 'p1_distance_run', 'p2_distance_run', 'rally_count', 'speed_mph']

# 设置目标变量为2维（comprehensive_momentum_1 和 comprehensive_momentum_2）
y = combined_df[['comprehensive_momentum_1', 'comprehensive_momentum_2']]

# 特征标准化
X = combined_df[feature_columns].copy()
for column in feature_columns:
    X[column].fillna(X[column].mean(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 数据拆分
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 选择模型
model = RandomForestRegressor()

# 训练模型
model.fit(X_train, y_train)

# 找到1301比赛的数据
match_id = "1301"
match_file = f'{folder_path}/2023-wimbledon-{match_id}.csv'
match_df = pd.read_csv(match_file)

# 使用相同的特征列和处理步骤
X_match = match_df[feature_columns].copy()
for column in feature_columns:
    X_match[column].fillna(X_match[column].mean(), inplace=True)

# 标准化
X_match_scaled = scaler.transform(X_match)

y_pred_match = model.predict(X_match_scaled)

# 找到交点的函数，并计算变化
def find_intersections_and_changes(y1, y2, threshold=4.5):
    intersections = []
    for i in range(1, len(y1)):
        change_1 = y1[i] - y1[i-1]
        change_2 = y2[i] - y2[i-1]
        total_change = abs(change_1) + abs(change_2)

        if total_change > threshold:
            if ((y1[i-1] > y2[i-1]) and (y1[i] < y2[i])) or ((y1[i-1] < y2[i-1]) and (y1[i] > y2[i])):
                intersection_x = i
                intersection_y = y1[i] + (y2[i] - y1[i]) / 2
                intersections.append((intersection_x, intersection_y, change_1, change_2))
    return intersections

intersections = find_intersections_and_changes(y_pred_match[:, 0], y_pred_match[:, 1])

# 打印交点位置和变化
for i, (x, y, change_1, change_2) in enumerate(intersections, 1):
    print(f"Intersection {i}: (x={x}, y={round(y, 2)}) - Change in Momentum 1: {change_1:.2f}, Change in Momentum 2: {change_2:.2f}")

# 可视化预测曲线和交点
plt.figure(figsize=(15, 6))
plt.plot(y_pred_match[:, 0], label='Predicted Comprehensive Momentum 1', alpha=0.7)
plt.plot(y_pred_match[:, 1], label='Predicted Comprehensive Momentum 2', alpha=0.7)
for x, y, _, _ in intersections:
    plt.scatter(x, y, color='black')
plt.title(f'Predicted Comprehensive Momentum 1 and 2 with Intersections (Match {match_id})')
plt.xlabel('Sample')
plt.ylabel('Comprehensive Momentum')
plt.legend()
plt.show()