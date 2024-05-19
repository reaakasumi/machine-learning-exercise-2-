import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np

#preprocessing
df = pd.read_csv("garments_worker_productivity.csv")
header = df.columns.tolist()
print(header)
df_scaled = preprocessing.MinMaxScaler().fit_transform(df.values)
# dataset = pd.DataFrame(df_scaled, columns=header)

#
# not_important = []
# for i in range(1, len(header)):
#     data = dataset[header[i]]
#     if (max(data) == min(data)):
#         not_important.append(header[i])
#
# dataset.drop(not_important, axis=1, inplace=True)
