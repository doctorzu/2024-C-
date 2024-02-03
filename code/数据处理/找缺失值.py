import pandas as pd

# 加载数据
file_path = 'Wimbledon_featured_matches.csv'
data = pd.read_csv(file_path)

# 找到每列的缺失值数量
missing_values = data.isnull().sum()

# 输出每列的缺失值数量
print("每列的缺失值数量:\n", missing_values)
