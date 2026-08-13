import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 0. Generate Synthetic Dataset
# ==========================================
print("--- GENERATING HOUSE PRICING DATASET ---")
np.random.seed(42)

# Features: Size (sq ft), Number of Bedrooms, Age of House (years)
# Target: Price ($)
n_samples = 200
size = np.random.randint(800, 4000, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
age = np.random.randint(0, 50, n_samples)

# Creating a linear relationship with some random noise
price = (size * 150) + (bedrooms * 10000) - (age * 500) + np.random.randint(-20000, 20000, n_samples)

df = pd.DataFrame({'Size': size, 'Bedrooms': bedrooms, 'Age': age, 'Price': price})
print(df.head())

# ==========================================
# PART 1: Simple Linear Regression
# ==========================================
print("\n" + "=" * 50)
print("--- PART 1: SIMPLE LINEAR REGRESSION ---")
# We will use only 'Size' to predict 'Price'

X_simple = df[['Size']]  # Independent variable (must be 2D array)
y_simple = df['Price']  # Dependent variable

# Split data into training and testing sets (80% train, 20% test)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)

# Initialize and train the model
simple_model = LinearRegression()
simple_model.fit(X_train_s, y_train_s)

# Make predictions on the test set
y_pred_s = simple_model.predict(X_test_s)

# ==========================================
# PART 2: Multiple Linear Regression
# ==========================================
print("\n" + "=" * 50)
print("--- PART 2: MULTIPLE LINEAR REGRESSION ---")
# We will use 'Size', 'Bedrooms', and 'Age' to predict 'Price'

X_multi = df[['Size', 'Bedrooms', 'Age']]
y_multi = df['Price']

# Split data
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# Initialize and train the model
multi_model = LinearRegression()
multi_model.fit(X_train_m, y_train_m)

# Make predictions
y_pred_m = multi_model.predict(X_test_m)

# ==========================================
# PART 3: Model Evaluation Metrics
# ==========================================
print("\n" + "=" * 50)
print("--- PART 3: MODEL EVALUATION ---")


def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Evaluation Metrics for {model_name}:")
    print(f"Mean Squared Error (MSE):       {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE):      {mae:.2f}")
    print(f"R-squared (R2):                 {r2:.4f}\n")


evaluate_model(y_test_s, y_pred_s, "Simple Linear Regression (Size only)")
evaluate_model(y_test_m, y_pred_m, "Multiple Linear Regression (All Features)")

# ==========================================
# PART 4: Visualization (Simple Regression)
# ==========================================
plt.figure(figsize=(10, 6))
plt.scatter(X_test_s, y_test_s, color='blue', label='Actual Data', alpha=0.6)
plt.plot(X_test_s, y_pred_s, color='red', linewidth=2, label='Regression Line')
plt.title('Simple Linear Regression: Size vs Price')
plt.xlabel('Size (sq ft)')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()