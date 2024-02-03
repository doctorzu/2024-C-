import pandas as pd

# 加载数据
file_path = 'Wimbledon_featured_matches.csv'
data = pd.read_csv(file_path)

# 选出x列为缺失值的所有行
# 替换 'x' 为你要检查的列名
missing_x = data[data['serve_width'].isnull()]

# 保存这些行为一个新的CSV文件
output_file_path = 'missing_x_rows.csv'
missing_x.to_csv(output_file_path, index=False)

print(f"包含x列缺失值的行已保存到 {output_file_path}")
