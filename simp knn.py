import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold


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

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def get_evaluation(model, features, target):
    kf = KFold(n_splits=10)
    mse = []
    rmse = []
    corr = []
    for train_index, test_index in kf.split(features):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mse.append(mean_squared_error(y_test, pred))
        rmse.append(root_mean_squared_error(y_test, pred))
        correlation_matrix = np.corrcoef(y_test, pred)
        corr.append(correlation_matrix[0, 1])

    return {
        'MSE': np.mean(mse),
        'RMSE': np.mean(rmse),
        'Correlation': np.mean(corr)
    }

class custom_knn:
    def __init__(self):
        self.k = None
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train, k=5):
        self.k = k
        self.x_train = x_train
        # Ensure y_train is a numpy array for consistent indexing
        self.y_train = np.array(y_train)

    def predict(self, x_test, distance_metric=1):
        # Compute distances based on the specified metric
        if distance_metric == 1:
            distances = euclidean_distances(x_test, self.x_train)
        elif distance_metric == 2:
            distances = cosine_distances(x_test, self.x_train)
        elif distance_metric == 3:
            distances = manhattan_distances(x_test, self.x_train)
        else:
            print("Invalid distance metric")
            return None

        # Get indices of the k nearest neighbors
        k_indices = np.argsort(distances, axis=1)[:, :self.k]

        # Aggregate the nearest labels
        # `k_indices` is 2D, so we need to handle dimensions properly when indexing `y_train`
        k_nearest_labels = np.array([self.y_train[k_index] for k_index in k_indices])

        # Return the mean of the nearest labels
        return np.mean(k_nearest_labels, axis=1)

# Initialize and fit the KNN model
knn_model = custom_knn()
knn_model.fit(X_train[numeric_cols], y_train, k=5)

# Make predictions using different distance metrics
y_pred_euclidean = knn_model.predict(X_test[numeric_cols], distance_metric=1)
y_pred_cosine = knn_model.predict(X_test[numeric_cols], distance_metric=2)
y_pred_manhattan = knn_model.predict(X_test[numeric_cols], distance_metric=3)

# Initialize the KNeighborsRegressor from scikit-learn
sklearn_knn = KNeighborsRegressor(n_neighbors=5, metric='euclidean')

# Fit the model on the training data
sklearn_knn.fit(X_train[numeric_cols], y_train)

# Make predictions on the test data
y_pred_sklearn = sklearn_knn.predict(X_test[numeric_cols])

# Set up the figure and axes
plt.figure(figsize=(14, 7))
plt.title('Comparison of KNN Predictions with Actual Data')
plt.xlabel('Actual Productivity')
plt.ylabel('Predicted Productivity')
plt.grid(True)

# Actual vs. Predicted scatter plot for each model
plt.scatter(y_test, y_pred_euclidean, color='blue', alpha=0.5, label='Custom KNN - Euclidean')
plt.scatter(y_test, y_pred_cosine, color='red', alpha=0.5, label='Custom KNN - Cosine')
plt.scatter(y_test, y_pred_manhattan, color='green', alpha=0.5, label='Custom KNN - Manhattan')
plt.scatter(y_test, y_pred_sklearn, color='purple', alpha=0.5, label='Sklearn KNN - Euclidean')

# Ideal line where predicted equals actual
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

plt.legend()

plt.show()

evaluation_sklearn = get_evaluation(sklearn_knn, X[numeric_cols], y)

evaluation_custom = get_evaluation(knn_model, X[numeric_cols], y)

print("Evaluation for Sklearn KNN:")
print(evaluation_sklearn)

print("Evaluation for Custom KNN:")
print(evaluation_custom)
