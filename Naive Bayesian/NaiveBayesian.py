import numpy as np
import pandas as pd

# read CSV
df_data = pd.read_csv('data.csv')
# print(df_data["temperature"])
# print(df_data["temperature"].nunique())

# # 使用布尔索引筛选出特定值的行
# filtered_df = df_data[df_data["temperature"] == 'warm']
# # print(filtered_df)
#
#
# features = df_data.columns.values.tolist()
# # print(features)
# frequency_feature = df_data[features[0]].value_counts()
# print(frequency_feature)
# frequency_feature_value = df_data[features[0]].value_counts().get('sunny', 0)
# print(frequency_feature_value)



class NaiveBayes():
    def __init__(self, data):
        self.data = data
        self.features = data.columns.values.tolist()
        pass

    def Multinomial(self, category, feature, value, k=1): # k is smooth parameter
        data = self.data[self.data[self.features[-1]] == category]
        n = data[feature].nunique()
        N_y = data[feature].count()
        N_yi = data[feature].value_counts().get(value, 0)
        return (N_yi+k)/(N_y+k*n)



    def Gaussian(self):
        pass

if __name__ == '__main__':
    nai = NaiveBayes(df_data)
    result = nai.Multinomial('yes', 'humidity', 'high')
    sample = ["sunny", 'cold', 'medium', 'low']
    predict_yes, predict_no = 0, 0
    for category, feature in zip(nai.features[:-1], sample):
        predict_yes += nai.Multinomial('yes', category, feature)
        predict_no += nai.Multinomial('no', category, feature)
    if predict_yes >= predict_no:
        print('Yes')
    else:
        print('No')