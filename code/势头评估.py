import pandas as pd
import matplotlib.pyplot as plt

path = 'Wimbledon_featured_matches.csv'
df = pd.read_csv(path)
# df.info()
target_match_id = '2023-wimbledon-1701'
filtered_data = df[df['match_id'] == target_match_id].copy()

"""
计算综合势头得分函数
"""
def calculate_comprehensive_momentum(data, player_number, window_size=4):
    momentum_scores = [0] * len(data)
    consecutive_point_wins = 0  # 追踪连续得分
    consecutive_game_wins = 0  # 追踪连续获胜的局数
    previous_game_winner = None  # 追踪上一局的获胜者
    initial_break_point_value = 1  # 破发的基础势头得分增加值

    for i in range(1, len(data)):
        recent_data = data[max(0, i - window_size):i]
        momentum_score = 0

        for _, feature in recent_data.iterrows():
            # 基本的势头得分计算
            P_t = 1 if feature['point_victor'] == player_number else -1
            S_t = 1.2 if feature['server'] == player_number else 1.0
            base_momentum = P_t * S_t
            momentum_score += base_momentum
            break_point_value = initial_break_point_value  # 重置破发得分值

            # 连续得分补正（线性）
            if P_t == 1:
                consecutive_point_wins += 1
            else:
                consecutive_point_wins = 0  # 在失分时重置
            momentum_score += 0.03 * consecutive_point_wins  # 每连续获胜增加额外得分

            # 连续小局获胜补正（线性）
            if feature['game_victor']:
                current_game_winner = feature['game_victor']
                if current_game_winner == player_number:
                    if current_game_winner == previous_game_winner:
                        consecutive_game_wins += 1
                    else:
                        consecutive_game_wins = 0  # 重置连续获胜局数
                previous_game_winner = current_game_winner
                momentum_score += 0.2 * consecutive_game_wins  # 连续获胜局数的影响

            # 大比分差距补正（指数）
            if feature['set_victor']:
                player1_set = feature['p1_sets'] + 1 if feature['set_victor'] == player_number else feature['p1_sets']
                player2_set = feature['p2_sets'] + 1 if feature['set_victor'] == player_number else feature['p2_sets']
                diff = (player2_set - player1_set) * (-1 ** player_number)  # player1为-1， player2为+1
                momentum_score += 0.1 * (2 ** diff)

            # 小比分差距补正（线性）
            if feature['game_victor']:
                score_diff = abs(feature['p1_games'] - feature['p2_games'])
                momentum_score += 0.02 * score_diff * P_t

            # 错失破发点对破发的势头得分增加值的削弱
            if feature['p1_break_pt_missed'] == 1 or feature['p2_break_pt_missed'] == 1:
                break_point_value -= 0.1  # 削弱的权值

            # (被)破发的影响
            if feature['p1_break_pt_won'] == 1 or feature['p2_break_pt_won'] == 1:
                break_point_value = max(break_point_value, 0.1)
                momentum_score += break_point_value * P_t

            # 拍数和跑动距离的影响
            rally_factor = feature['rally_count'] / 30  # 归一化回合数
            distance_factor = (feature['p1_distance_run'] + feature['p2_distance_run']) / 122  # 归一化跑动距离
            momentum_score += 2.0 * rally_factor * distance_factor * P_t

        momentum_scores[i] = momentum_score

    return momentum_scores

"""
为两位球员计算综合势头得分
"""
filtered_data['comprehensive_momentum_1'] = calculate_comprehensive_momentum(filtered_data, player_number=1)
filtered_data['comprehensive_momentum_2'] = calculate_comprehensive_momentum(filtered_data, player_number=2)

# Define the threshold for a significant momentum shift
threshold = 4

# Initialize lists to store the points of positive and negative shifts for both players
shifts_player_1 = []
shifts_player_2 = []

# Calculate the momentum change for each point and find shifts
for i in range(1, len(filtered_data)):
    change_1 = filtered_data['comprehensive_momentum_1'].iloc[i] - filtered_data['comprehensive_momentum_1'].iloc[i - 1]
    change_2 = filtered_data['comprehensive_momentum_2'].iloc[i] - filtered_data['comprehensive_momentum_2'].iloc[i - 1]

    if abs(change_1) >= threshold:
        shift_type = 'Positive' if change_1 > 0 else 'Negative'
        shifts_player_1.append((i, shift_type, change_1))
    if abs(change_2) >= threshold:
        shift_type = 'Positive' if change_2 > 0 else 'Negative'
        shifts_player_2.append((i, shift_type, change_2))

# Annotation
for point, shift_type, change in shifts_player_1:
    set_no = filtered_data['set_no'].iloc[point]
    game_no = filtered_data['game_no'].iloc[point]
    print(
        f"Player 1 had a {shift_type} shift at point number {point}, during set {set_no}, game {game_no}. GET {change}")

for point, shift_type, change in shifts_player_2:
    set_no = filtered_data['set_no'].iloc[point]
    game_no = filtered_data['game_no'].iloc[point]
    print(
        f"Player 2 had a {shift_type} shift at point number {point}, during set {set_no}, game {game_no}. GET {change}")

# 筛选出破发成功的点
break_points_won_1 = filtered_data[filtered_data['p1_break_pt_won'] == 1]
break_points_won_2 = filtered_data[filtered_data['p2_break_pt_won'] == 1]

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(filtered_data['point_no'], filtered_data['comprehensive_momentum_1'], label='Player 1 Momentum', color='blue')
plt.plot(filtered_data['point_no'], filtered_data['comprehensive_momentum_2'], label='Player 2 Momentum', color='red')

for point, shift_type, _ in shifts_player_1:
    marker = '*' if shift_type == 'Positive' else 'x'
    color = 'green' if shift_type == 'Positive' else 'red'
    plt.scatter(filtered_data['point_no'].iloc[point], filtered_data['comprehensive_momentum_1'].iloc[point],
                color=color, marker=marker, s=100)

for point, shift_type, _ in shifts_player_2:
    marker = '*' if shift_type == 'Positive' else 'x'
    color = 'green' if shift_type == 'Positive' else 'red'
    plt.scatter(filtered_data['point_no'].iloc[point], filtered_data['comprehensive_momentum_1'].iloc[point],
                color=color, marker=marker, s=100)

# 用黑色点标记破发成功的点
plt.scatter(break_points_won_1['point_no'], break_points_won_1['comprehensive_momentum_1'], color='black', marker='o', label='Player 1 Break Points Won')
plt.scatter(break_points_won_2['point_no'], break_points_won_2['comprehensive_momentum_2'], color='black', marker='s', label='Player 2 Break Points Won')

# Add labels and title to the plot
plt.xlabel('Point Number')
plt.ylabel('Momentum Score')
plt.title('Advanced Momentum Score with Consecutive Points Comparison Throughout the Match')
plt.legend()
plt.show()
