import pandas as pd

# 替换为您的文件路径
file_path = 'Wimbledon_featured_matches.csv'

try:
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 确保 'event' 列存在
    if 'serve_width' in df.columns:
        # 获取 'event' 列的所有唯一值
        unique_events = df['serve_width'].unique()

        # 打印所有唯一值
        for event in unique_events:
            print(event)

        # 打印唯一值的个数
        print(f"唯一值的个数: {len(unique_events)}")
    else:
        print("没有找到 'club' 列")
except FileNotFoundError:
    print(f"文件 {file_path} 未找到")
except Exception as e:
    print(f"处理文件时发生错误: {e}")
