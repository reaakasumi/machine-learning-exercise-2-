import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDRegressor
from gradient_descent import gradient_descent
from simp_knn import custom_knn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

def get_evaluation(model, features, target):
    kf = KFold(n_splits=10)
    mse = []
    rmse = []
    corr = []
    for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = target.to_numpy()[train_index], target.to_numpy()[test_index]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mse.append(mean_squared_error(y_test.flatten(), pred))
        rmse.append(root_mean_squared_error(y_test.flatten(), pred))
        correlation_matrix = np.corrcoef(np.array(y_test.flatten()), np.array(pred))
        corr.append(correlation_matrix[0, 1])

    mse = sum(mse) / len(mse)
    rmse = sum(rmse) / len(rmse)
    corr = sum(corr) / len(corr)
    return mse, rmse, corr


#preprocessing
df = pd.read_csv("Student_Performance.csv")
features = df. loc[:, df. columns != "Performance Index"]
target = df. loc[:, df. columns == "Performance Index"]
features['Extracurricular Activities'] = features['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

scalar = MinMaxScaler()
features = scalar.fit_transform(features)


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=2)

X_train_minmax = X_train
X_test_minmax = X_test
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
model = gradient_descent(0.00001, 10000)
model.fit(X_train_minmax,y_train)
pred = model.predict(X_test_minmax)

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(features, target, test_size=0.25, random_state=2)
# Using sklearn
sk_model = SGDRegressor()
sk_model.fit(X_train_2,y_train_2)
sk_model_pred = sk_model.predict(X_test_2)

result = pd.DataFrame({'Actual': y_test.values.flatten(), 'sk-learn': sk_model_pred.flatten(), "Custom": pred.flatten()})
print(result)
print(sk_model.coef_)
print(model.get_weights())




"""
print("GRADIENT DESCENT")
custom_gd_eval = get_evaluation(gradient_descent(0.00001, 1000), features, target)
sklearn_gd_eval = get_evaluation(SGDRegressor(), features, target)
"""
print("KNN")
custom_knn_eval = get_evaluation(custom_knn(11), features, target)
sklearn_knn_eval = get_evaluation(KNeighborsRegressor(n_neighbors=11), features, target)


#result -> both are similar!
