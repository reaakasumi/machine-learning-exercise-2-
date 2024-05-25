import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

df = pd.read_csv("garments_worker_productivity.csv")
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek

df = pd.get_dummies(df, columns=['quarter', 'department', 'day'])

df.dropna(inplace=True)

X = df.drop('actual_productivity', axis=1)
y = df['actual_productivity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = X_train.copy()
train_data['actual_productivity'] = y_train

correlation_matrix = train_data.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

print(correlation_matrix['actual_productivity'].sort_values(ascending=False))

# Scaling selected columns
scaler = StandardScaler()
columns_to_scale = ['targeted_productivity', 'smv', 'over_time', 'incentive', 'idle_time', 'no_of_workers']
X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

# Ensure to scale only numeric columns
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Add a constant to the model (needed for VIF calculation)
X_train_const = sm.add_constant(X_train[numeric_cols])  # Include only numeric columns

# Calculate VIF
vifs = pd.DataFrame()
vifs["VIF"] = [variance_inflation_factor(X_train_const.values, i) for i in range(X_train_const.shape[1])]
vifs["features"] = X_train_const.columns
print(vifs)

# Removing outliers based on z-scores in the training set
z_scores = np.abs(stats.zscore(X_train[columns_to_scale]))
X_train = X_train[(z_scores < 3).all(axis=1)]
y_train = y_train[X_train.index]  # Ensure y_train is also filtered

# Selecting one feature for simple linear regression via gradient descent
x_feature = 'incentive'
x_train = X_train[[x_feature]]  # Use double brackets to keep DataFrame structure
x_test = X_test[[x_feature]]

# Initialize GradientDescent class
class GradientDescent:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.epochs = epochs
        self.alpha = learning_rate
        self.weights = None

    def compute_rss(self, x, y):
        predictions = np.dot(x, self.weights)
        return np.sum((y - predictions) ** 2)

    def derivative_rss(self, x, y):
        predictions = np.dot(x, self.weights)
        errors = y - predictions
        return -2 * np.dot(x.T, errors)

    def fit(self, x_train, y_train):
        self.weights = np.random.randn(x_train.shape[1])
        for i in range(self.epochs):
            step = self.derivative_rss(x_train, y_train) / len(x_train)
            self.weights = self.weights - self.alpha * step
            if i % 100 == 0:
                rss = self.compute_rss(x_train, y_train)
                print(f"Iteration {i}: weights = {self.weights}, RSS = {rss}")

    def predict(self, x_test):
        return np.dot(x_test, self.weights)

# Prepare data with intercept term for GradientDescent
x_train = np.c_[np.ones(x_train.shape[0]), x_train]
x_test = np.c_[np.ones(x_test.shape[0]), x_test]

# Experiment with different learning rates
learning_rates = [0.001, 0.01, 0.1]
results = {}
colors = ['blue', 'orange', 'purple']

for lr, color in zip(learning_rates, colors):
    gd_model = GradientDescent(learning_rate=lr, epochs=1000)
    gd_model.fit(x_train, y_train)
    y_pred_custom_gd = gd_model.predict(x_train)
    results[lr] = (y_pred_custom_gd, color)

# Using scikit-learn's SGDRegressor for comparison
sgd_regressor = SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.01, learning_rate='constant', random_state=42)
sgd_regressor.fit(x_train[:, 1:], y_train)  # exclude the added constant term for sklearn

# Predictions for plotting
y_pred_sgd = sgd_regressor.predict(x_train[:, 1:])

# Plotting the results
plt.scatter(x_train[:, 1:], y_train, color='blue', label='Actual Data')
for lr, (y_pred, color) in results.items():
    plt.plot(x_train[:, 1:], y_pred, color=color, label=f'Custom GD Prediction (lr={lr})')
plt.plot(x_train[:, 1:], y_pred_sgd, color='green', label='SGD Prediction')
plt.xlabel('Scaled ' + x_feature)
plt.ylabel('Actual Productivity')
plt.title('Fit of Linear Model using Gradient Descent with Different Learning Rates')
plt.legend()
plt.show()
