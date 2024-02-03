import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
file_path = 'Wimbledon_featured_matches.csv'
data = pd.read_csv(file_path)

# 获取独特的比赛序号
unique_matches = data['match_id'].unique()

for match in unique_matches:
    # 获取当前比赛的数据
    match_data = data[data['match_id'] == match]['speed_mph'].dropna()  # 删除NaN值

    # 计算离群值
    Q1 = match_data.quantile(0.25)
    Q3 = match_data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = match_data[(match_data < (Q1 - 1.5 * IQR)) | (match_data > (Q3 + 1.5 * IQR))]

    # 输出离群值个数和总个数
    print(f"比赛 {match} 的发球速度离群值个数: {len(outliers)}")
    print(f"比赛 {match} 的发球速度总个数: {len(match_data)}\n")


for match in unique_matches:
    # 获取当前比赛的数据
    match_data = data[data['match_id'] == match]['speed_mph'].dropna()  # 删除NaN值

    # 绘制箱型图
    plt.figure(figsize=(10, 6))
    plt.boxplot(match_data, patch_artist=True)

    # 标注平均值
    mean_value = match_data.mean()
    plt.scatter(1, mean_value, color='red', zorder=3, label=f'平均值: {mean_value:.2f}')

    # 添加标题和标签
    plt.title(f'比赛 {match} 中的发球速度')
    plt.ylabel('发球速度 (speed_mph列)')

    # 显示图例
    plt.legend()

    # 显示图表
    plt.show()


