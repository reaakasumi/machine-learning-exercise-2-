import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from gradient_descent import gradient_descent
from simp_knn import custom_knn
from sklearn.preprocessing import MinMaxScaler

#preprocessing
df = pd.read_csv("Student_Performance.csv").head(100)
#df.insert(0, "b", 1) #yx + b
features = df. loc[:, df. columns != "Performance Index"]
target = df. loc[:, df. columns == "Performance Index"]
features['Extracurricular Activities'] = features['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

scalar = MinMaxScaler()
features = scalar.fit_transform(features)
#features[::, 0] = 1.

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=2)

model = custom_knn()
model.fit(X_train,y_train, 2)

pred = model.predict(X_test, 3)
sk_result = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': pred.flatten()})
print(sk_result)
""" 
model = gradient_descent()
model.fit(X_train,y_train)
pred = model.predict(X_test)
result = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': pred.flatten()})
print(result)

# Using sklearn
sk_model = SGDRegressor()
sk_model.fit(X_train,y_train)
sk_model_pred = sk_model.predict(X_test)
sk_result = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': sk_model_pred.flatten()})
print(sk_result)

#result -> both are similar!

"""