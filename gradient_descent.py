import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Load and preprocess the dataset
df = pd.read_csv("garments_worker_productivity.csv")
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek

# Encoding categorical variables
df = pd.get_dummies(df, columns=['quarter', 'department', 'day'])

# Scaling selected columns
scaler = StandardScaler()
columns_to_scale = ['targeted_productivity', 'smv', 'over_time', 'incentive', 'idle_time', 'no_of_workers']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Removing outliers
df = df[(np.abs(stats.zscore(df[columns_to_scale])) < 3).all(axis=1)]

# Train-test split
X = df.drop('actual_productivity', axis=1)
y = df['actual_productivity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Selecting one feature for simple linear regression via gradient descent
x_feature = 'no_of_workers'
x = X_train[x_feature].values.reshape(-1, 1)
y = y_train.values

# Normalize the feature manually before applying gradient descent
scaler_x = StandardScaler()
x_scaled = scaler_x.fit_transform(x)

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
        print(f"Final model: y = {self.weights[0]} * x")

    def predict(self, x_test):
        return np.dot(x_test, self.weights)

# Prepare data with intercept term for GradientDescent
x_scaled_intercept = np.c_[np.ones(x_scaled.shape[0]), x_scaled]

# Initialize and fit custom gradient descent model
gd_model = GradientDescent(learning_rate=0.01, epochs=1000)
gd_model.fit(x_scaled_intercept, y)

# Predictions for plotting
y_pred_custom_gd = gd_model.predict(x_scaled_intercept)

# Using scikit-learn's SGDRegressor for comparison
sgd_regressor = SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.01, learning_rate='constant', random_state=42)
sgd_regressor.fit(x_scaled, y)

# Predictions for plotting
y_pred_sgd = sgd_regressor.predict(x_scaled)

# Plotting the results
plt.scatter(x_scaled, y_train, color='blue', label='Actual Data')
plt.plot(x_scaled, y_pred_custom_gd, color='red', label='Custom GD Prediction')
plt.plot(x_scaled, y_pred_sgd, color='green', label='SGD Prediction')
plt.xlabel('Scaled Number of Workers')
plt.ylabel('Actual Productivity')
plt.title('Fit of Linear Model using Gradient Descent')
plt.legend()
plt.show()
