import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
file_path = 'Wimbledon_featured_matches.csv'
data = pd.read_csv(file_path)

# 确保R列不为0
data = data[data['rally_count'] != 0]
data = data[data['p1_distance_run'] != 0]
data = data[data['p2_distance_run'] != 0]

# 计算平均跑动距离
data['P1_avg_distance'] = data['p1_distance_run'] / data['rally_count']
data['P2_avg_distance'] = data['p2_distance_run'] / data['rally_count']

# 计算每场比赛每个球员的离群值个数和总个数
unique_matches = data['match_id'].unique()
for match in unique_matches:
    match_data = data[data['match_id'] == match]

    # 计算球员1和球员2的离群值
    Q1_P1 = match_data['P1_avg_distance'].quantile(0.25)
    Q3_P1 = match_data['P1_avg_distance'].quantile(0.75)
    IQR_P1 = Q3_P1 - Q1_P1
    outliers_P1 = match_data[((match_data['P1_avg_distance'] < (Q1_P1 - 1.5 * IQR_P1)) | (
                match_data['P1_avg_distance'] > (Q3_P1 + 1.5 * IQR_P1)))]

    Q1_P2 = match_data['P2_avg_distance'].quantile(0.25)
    Q3_P2 = match_data['P2_avg_distance'].quantile(0.75)
    IQR_P2 = Q3_P2 - Q1_P2
    outliers_P2 = match_data[((match_data['P2_avg_distance'] < (Q1_P2 - 1.5 * IQR_P2)) | (
                match_data['P2_avg_distance'] > (Q3_P2 + 1.5 * IQR_P2)))]

    # 输出结果
    print(f"比赛 {match}:")
    print(f"球员1的离群值个数: {outliers_P1.shape[0]}, 总个数: {match_data['P1_avg_distance'].count()}")
    print(f"球员2的离群值个数: {outliers_P2.shape[0]}, 总个数: {match_data['P2_avg_distance'].count()}\n")


# 对每场比赛生成箱型图
unique_matches = data['match_id'].unique()
for match in unique_matches:
    match_data = data[data['match_id'] == match]

    plt.figure(figsize=(12, 8))
    box = plt.boxplot([match_data['P1_avg_distance'].dropna(), match_data['P2_avg_distance'].dropna()], labels=['球员1', '球员2'], patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'), whiskerprops=dict(color='blue'), capprops=dict(color='blue'))

    # 设置标题和标签
    plt.title(f'比赛 {match} 中球员平均每次击球的跑动距离', fontsize=16, fontweight='bold')
    plt.ylabel('平均跑动距离', fontsize=14)
    plt.xlabel('球员', fontsize=14)

    # 标出平均值
    mean_p1 = match_data['P1_avg_distance'].mean()
    mean_p2 = match_data['P2_avg_distance'].mean()
    plt.scatter([1, 2], [mean_p1, mean_p2], color='red', zorder=3, label='平均值')

    # 添加格网线
    plt.grid(True)

    plt.legend()
    plt.show()