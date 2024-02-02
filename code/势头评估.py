import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
data_file_path = 'Wimbledon_featured_matches.csv'
df = pd.read_csv(data_file_path)

# 根据特定的比赛ID筛选数据，并创建副本以避免SettingWithCopyWarning
target_match_id = '2023-wimbledon-1701'
filtered_data = df[df['match_id'] == target_match_id].copy()


# 定义一个计算综合势头得分的函数
def calculate_comprehensive_momentum(data, player_number, window_size=5):
    momentum_scores = [0] * len(data)
    consecutive_wins = 0  # 追踪连续获胜

    for i in range(len(data)):
        recent_data = data[max(0, i - window_size):i + 1]
        momentum_score = 0
        for _, row in recent_data.iterrows():
            # 基本的势头得分计算
            P_t = 1 if row['point_victor'] == player_number else -1
            S_t = 1.2 if row['server'] == player_number else 1.0
            base_momentum = P_t * S_t

            # 确保基础势头得分为正值，如果球员赢得得分点
            if P_t == 1:
                base_momentum = max(base_momentum, 0.5)  # 至少增加 0.5 的势头得分

            momentum_score += base_momentum

            # 连续得分的调整
            if P_t == 1:
                consecutive_wins += 1
            else:
                consecutive_wins = 0  # 在失分时重置
            momentum_score += 0.2 * consecutive_wins  # 每连续获胜增加额外得分

            # 破发点的影响
            if (player_number == 1 and row['p1_break_pt'] > 0) or (player_number == 2 and row['p2_break_pt'] > 0):
                momentum_score += 0.5 * P_t

            # 持续时间和跑动距离的影响
            rally_factor = row['rally_count'] / 10  # 归一化回合数
            distance_factor = (row['p1_distance_run'] + row['p2_distance_run']) / 50  # 归一化跑动距离
            momentum_score += rally_factor * distance_factor * P_t

            # 比分差距的影响
            score_diff = abs(row['p1_games'] - row['p2_games'])
            momentum_score += 0.2 * score_diff * P_t

        momentum_scores[i] = momentum_score

    return momentum_scores


# 为两位球员计算综合势头得分
filtered_data['comprehensive_momentum_1'] = calculate_comprehensive_momentum(filtered_data, player_number=1)
filtered_data['comprehensive_momentum_2'] = calculate_comprehensive_momentum(filtered_data, player_number=2)

# 绘制综合势头得分
plt.figure(figsize=(12, 6))
plt.plot(filtered_data['point_no'], filtered_data['comprehensive_momentum_1'], label='Player 1 Momentum', color='blue')
plt.plot(filtered_data['point_no'], filtered_data['comprehensive_momentum_2'], label='Player 2 Momentum', color='red')
plt.xlabel('Point Number')
plt.ylabel('Advanced Momentum Score with Consecutive Points')
plt.title('Advanced Momentum Score with Consecutive Points Comparison Throughout the Match')
plt.legend()
plt.show()


