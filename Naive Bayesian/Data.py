import pandas as pd

# create a simple DataFrame, it is our data
data = {
    'weather': ["sunny", "cloudy", "rainy", "sunny", "sunny", "cloudy", "rainy"],
    'temperature': ['hot', 'warm', 'cold', 'warm', 'hot', 'hot', 'warm'],
    'humidity': ['high', 'high', 'high', 'low', 'medium', 'medium', 'high'],
    'windy': ['low', 'high', 'high', 'low', 'low', 'high', 'low'],
    'is_sport': ['yes', 'yes', 'no', 'yes', 'no', 'yes', 'no'],     # we suppose that 'is_sport' is result
}
df = pd.DataFrame(data)

# 将DataFrame保存为CSV文件
df.to_csv('data.csv', index=False)  # index=False表示不保存行索引