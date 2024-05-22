import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from gradient_descent import gradient_descent
from sklearn.preprocessing import MinMaxScaler

#preprocessing
df = pd.read_csv("Student_Performance.csv").head(1000)
df.insert(0, "b", 1) #yx + b
features = df. loc[:, df. columns != "Performance Index"]
target = df. loc[:, df. columns == "Performance Index"]
features['Extracurricular Activities'] = features['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

scalar = MinMaxScaler()
features = scalar.fit_transform(features)
features[::, 0] = 1.

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = gradient_descent()
model.fit(X_train,y_train)
pred = model.predict(X_test)
result = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': pred.flatten()})
print(result)

#header = df.columns.tolist()
#print(header)
#df_scaled = preprocessing.MinMaxScaler().fit_transform(df.values)
# dataset = pd.DataFrame(df_scaled, columns=header)

#
# not_important = []
# for i in range(1, len(header)):
#     data = dataset[header[i]]
#     if (max(data) == min(data)):
#         not_important.append(header[i])
#
# dataset.drop(not_important, axis=1, inplace=True)
