import pandas as pd

# 加载数据
file_path = 'Wimbledon_featured_matches.csv'
data = pd.read_csv(file_path)

# 定义你想要检查的列名
column_zero = 'p2_distance_run'       # 替换为需要检查为0的列名
column_non_zero = 'p1_distance_run' # 替换为需要检查不为0的列名

column_zero2 = 'rally_count'

# 找出一列为0而另一列不为0的行
specified_rows = data[data[column_zero2] == 0]

# 获取这些行的数量
specified_rows_count = specified_rows.shape[0]

# 输出数量和行详情
print(f"一列为0而另一列不为0的行数量: {specified_rows_count}")
print("这些行的详细信息:\n", specified_rows)


# 保存这些行为一个新的CSV文件
output_file_path = 'zero_values_rows.csv'
specified_rows.to_csv(output_file_path, index=False)

print(f"所有指定列同时为0的行已保存到 {output_file_path}")