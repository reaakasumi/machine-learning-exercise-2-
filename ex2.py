import pandas as pd
from sklearn.preprocessing import StandardScaler
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
x = X_train[x_feature].values
y = y_train.values

# Initial guess for w1 (slope)
w1 = 0

# Learning rate
alpha = 0.001

# Function to compute the Residual Sum of Squares (RSS)
def compute_rss(w1, x, y):
    predictions = w1 * x
    return np.sum((y - predictions) ** 2)

# Function to compute the derivative of RSS with respect to w1
def derivative_rss_w1(w1, x, y):
    predictions = w1 * x
    errors = y - predictions
    return -2 * np.sum(errors * x)

# Number of iterations
iterations = 1000

# Gradient Descent Algorithm
for i in range(iterations):
    grad = derivative_rss_w1(w1, x, y)
    w1 = w1 - alpha * grad
    if i % 100 == 0:
        rss = compute_rss(w1, x, y)
        print(f"Iteration {i}: w1 = {w1}, RSS = {rss}")

print(f"Final model: y = {w1} * x")

# Plotting to visualize
plt.scatter(X_train[x_feature], y_train, color='blue')
plt.plot(X_train[x_feature], w1 * X_train[x_feature], color='red')
plt.xlabel('Scaled Number of Workers')
plt.ylabel('Actual Productivity')
plt.title('Fit of Linear Model using Gradient Descent')
plt.show()
