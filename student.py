import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from gradient_descent import gradient_descent
from simp_knn import custom_knn
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor

#preprocessing
df = pd.read_csv("Student_Performance.csv")
features = df. loc[:, df. columns != "Performance Index"]
target = df. loc[:, df. columns == "Performance Index"]
features['Extracurricular Activities'] = features['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=2)

scalar = MinMaxScaler()
X_train_minmax = scalar.fit_transform(X_train)
X_test_minmax = scalar.transform(X_test)

"""
## KNN

# Using custom knn
model = custom_knn()
model.fit(X_train_minmax,y_train, 5)
pred = model.predict(X_test_minmax, 3)


# Using sklearn
sk_model = KNeighborsRegressor(n_neighbors=5, metric='manhattan')
sk_model.fit(X_train_minmax,y_train)
sk_model_pred = sk_model.predict(X_test_minmax)


result = pd.DataFrame({'Actual': y_test.values.flatten(), 'sk-learn': sk_model_pred.flatten(), "Custom": pred.flatten()})
print(result)
"""

## Gradient Descent - Linear Regression

# Using custom gradient descent
model = gradient_descent()
model.fit(X_train_minmax,y_train)
pred = model.predict(X_test_minmax)

# Using sklearn
sk_model = SGDRegressor()
sk_model.fit(X_train_minmax,y_train)
sk_model_pred = sk_model.predict(X_test_minmax)

result = pd.DataFrame({'Actual': y_test.values.flatten(), 'sk-learn': sk_model_pred.flatten(), "Custom": pred.flatten()})
print(result)

#result -> both are similar!
