import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 0. Dataset Generation
# ==========================================
print("--- GENERATING DATASET ---")
np.random.seed(42)

# Creating a dataset with 150 records and 7 features
data = {
    'Age': np.random.randint(22, 60, 150),
    'Salary': np.random.randint(40000, 120000, 150),
    'Experience_Years': np.random.randint(0, 35, 150),
    'Department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], 150),
    'Join_Date': pd.date_range(start='2015-01-01', periods=150, freq='W'), # Weekly join dates
    'Performance_Score': np.random.uniform(1.0, 5.0, 150).round(2),
    'Target_Bonus': np.random.randint(1000, 10000, 150) # The target variable we want to predict
}
df = pd.DataFrame(data)

print("\nOriginal Dataset (First 5 Rows):")
print(df.head())


# ==========================================
# PART A: Feature Engineering
# ==========================================
print("\n" + "="*50)
print("--- PART A: FEATURE ENGINEERING ---")

# Technique 1: Date/Time Transformation
# Purpose: Machine learning models cannot read raw datetime objects. We convert it into a numeric 'Tenure' feature.
df['Tenure_Days'] = (pd.to_datetime('2024-01-01') - df['Join_Date']).dt.days

# Technique 2: Binning / Discretization
# Purpose: Groups continuous age data into distinct categories, helping the model find patterns across age brackets.
df['Age_Group'] = pd.cut(df['Age'], bins=[20, 35, 50, 65], labels=['Junior', 'Mid-Level', 'Senior'])

# Technique 3: One-Hot Encoding
# Purpose: Converts categorical text data (Department and Age_Group) into numerical 0/1 columns so the algorithm can process them.
# drop_first=True prevents the dummy variable trap (multicollinearity).
df = pd.get_dummies(df, columns=['Department', 'Age_Group'], drop_first=True)

# Drop the original Join_Date column as it's no longer needed
df.drop('Join_Date', axis=1, inplace=True)

print("\nDataset After Feature Engineering (First 5 Rows):")
print(df.head())


# ==========================================
# PART B: Feature Selection
# ==========================================
print("\n" + "="*50)
print("--- PART B: FEATURE SELECTION ---")

# Technique 1: Correlation Analysis
# Purpose: Removes features that have a very low correlation with the target variable, or features that are highly correlated with each other (redundant).
print("\n1. Correlation Analysis:")
correlation_matrix = df.corr()
target_correlation = correlation_matrix['Target_Bonus'].sort_values(ascending=False)
print("Correlation with Target_Bonus:")
print(target_correlation)

# Visualizing Correlation Matrix (Optional but recommended)
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Technique 2: SelectKBest (Statistical Selection)
# Purpose: Uses statistical tests (f_regression for continuous targets) to score features and select the 'k' absolute best ones.
print("\n2. SelectKBest (f_regression):")
X = df.drop('Target_Bonus', axis=1)
y = df['Target_Bonus']

# Select the top 4 features
selector = SelectKBest(score_func=f_regression, k=4)
X_selected = selector.fit_transform(X, y)

# Get the names of the selected features
selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = X.columns[selected_feature_indices]

print(f"Top 4 Features retained by SelectKBest: {list(selected_feature_names)}")
print("Justification: These features scored the highest on the F-statistic test, meaning they have the strongest linear relationship with the target variable (Target_Bonus). The rest were removed to reduce dimensionality and noise.")


# ==========================================
# PART C: Feature Scaling
# ==========================================
print("\n" + "="*50)
print("--- PART C: FEATURE SCALING ---")
# We will scale the original numeric features to prepare them for distance-based algorithms (like KNN or SVM).

features_to_scale = ['Age', 'Salary', 'Experience_Years', 'Performance_Score', 'Tenure_Days']

# Technique 1: StandardScaler
# Purpose: Scales features so they have a mean of 0 and a standard deviation of 1. Best when data follows a normal distribution.
print("\n1. StandardScaler:")
standard_scaler = StandardScaler()
df_standard_scaled = df.copy()
df_standard_scaled[features_to_scale] = standard_scaler.fit_transform(df[features_to_scale])
print(df_standard_scaled[features_to_scale].head())

# Technique 2: MinMaxScaler
# Purpose: Scales all features to be strictly between 0 and 1. Best when you need bounded data and don't assume a normal distribution.
print("\n2. MinMaxScaler:")
minmax_scaler = MinMaxScaler()
df_minmax_scaled = df.copy()
df_minmax_scaled[features_to_scale] = minmax_scaler.fit_transform(df[features_to_scale])
print(df_minmax_scaled[features_to_scale].head())


