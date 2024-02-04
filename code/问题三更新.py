import pandas as pd
import glob
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# 读取数据
folder_path = 'data_with_momentum'
file_list = glob.glob(f'{folder_path}/2023-wimbledon-*.csv')
dataframes = [pd.read_csv(file) for file in file_list]


# 数据拆分函数
def dataset_split(dataframes, ratio=0.2):
    total_dataframes = len(dataframes)
    num_test = int(total_dataframes * ratio)
    num_train = total_dataframes - num_test

    random.shuffle(dataframes)
    train_dfs = dataframes[:num_train]
    test_dfs = dataframes[num_train:]

    combined_train_df = pd.concat(train_dfs, ignore_index=True)
    combined_test_df = pd.concat(test_dfs, ignore_index=True)

    return combined_train_df, combined_test_df


combined_train_df, combined_test_df = dataset_split(dataframes, ratio=0.2)

# 特征选择
feature_columns = ['p1_sets', 'p2_sets', 'p1_games', 'p2_games', 'server', 'p1_ace', 'p2_ace', 'p1_double_fault',
                   'p2_double_fault', 'p1_unf_err', 'p2_unf_err', 'p1_break_pt_won', 'p2_break_pt_won',
                   'p1_break_pt_missed', 'p2_break_pt_missed', 'p1_distance_run', 'p2_distance_run', 'rally_count',
                   'speed_mph']

# 处理缺失值
for column in feature_columns:
    mean_value = combined_train_df[column].mean()
    combined_train_df[column] = combined_train_df[column].fillna(mean_value)
    combined_test_df[column] = combined_test_df[column].fillna(mean_value)

X_train = combined_train_df[feature_columns]
X_test = combined_test_df[feature_columns]

# 目标变量
y_train = combined_train_df[['comprehensive_momentum_1', 'comprehensive_momentum_2']]
y_test = combined_test_df[['comprehensive_momentum_1', 'comprehensive_momentum_2']]

# 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 模型训练
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 使用测试集进行预测
y_pred = model.predict(X_test)

# R² 分数
r2 = r2_score(y_test, y_pred)
print("R² score:", r2)


# 找到交点的函数
def find_intersections(y1, y2, threshold=3.5):
    intersections = []
    for i in range(1, len(y1)):
        change_1 = abs(y1[i] - y1[i - 1])
        change_2 = abs(y2[i] - y2[i - 1])
        total_change = change_1 + change_2

        if total_change > threshold:
            if ((y1[i - 1] > y2[i - 1]) and (y1[i] < y2[i])) or ((y1[i - 1] < y2[i - 1]) and (y1[i] > y2[i])):
                intersection_x = i
                intersection_y = y1[i] + (y2[i] - y1[i]) / 2
                intersections.append((intersection_x, intersection_y))
    return intersections


# 为每场比赛单独进行预测并绘制曲线
for file in file_list:
    df = pd.read_csv(file)

    # 创建特征列的副本来处理缺失值
    features_df = df[feature_columns].copy()
    features_df = features_df.apply(pd.to_numeric, errors='coerce')
    features_df.fillna(features_df.mean(), inplace=True)

    # 检查是否还有缺失值
    if features_df.isnull().values.any():
        print(f"Warning: Missing values detected in file {file}.")
        continue

    # 标准化
    X_match = scaler.transform(features_df)

    # 预测
    y_pred_match = model.predict(X_match)

    # 寻找交点
    intersections = find_intersections(y_pred_match[:, 0], y_pred_match[:, 1])

    # 打印交点信息和分数变化
    for i, (x, y) in enumerate(intersections, 1):
        change_1 = y_pred_match[x][0] - y_pred_match[x - 1][0]
        change_2 = y_pred_match[x][1] - y_pred_match[x - 1][1]
        print(f"Match: {os.path.basename(file)}")
        print(
            f"Intersection {i}: (x={x}, y={round(y, 2)}) - Change in Score: Player 1: {round(change_1, 2)}, Player 2: {round(change_2, 2)}")

    # 绘制曲线
    plt.figure(figsize=(15, 6))
    plt.plot(y_pred_match[:, 0], label='Predicted Comprehensive Momentum 1', alpha=0.7)
    plt.plot(y_pred_match[:, 1], label='Predicted Comprehensive Momentum 2', alpha=0.7)
    for x, y in intersections:
        plt.scatter(x, y, color='black')
    plt.title(f'Predicted Comprehensive Momentum for Match: {os.path.basename(file)}')
    plt.xlabel('Sample')
    plt.ylabel('Comprehensive Momentum')
    plt.legend()
    plt.show()

# 特征重要性
feature_importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# 可视化特征重要性
plt.figure(figsize=(12, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Predicting Comprehensive Momentum')
plt.gca().invert_yaxis()
plt.show()

