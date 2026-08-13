import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Load and Prepare the Dataset
# ==========================================
print("--- LOADING BREAST CANCER DATASET ---")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)  # 0 = Malignant, 1 = Benign

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Crucial for models like SVM, KNN, and Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Dataset Size: {X.shape[0]} rows, {X.shape[1]} features.")
print(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples\n")

# ==========================================
# 2. Initialize Classification Models
# ==========================================
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# ==========================================
# 3. Train, Predict, and Evaluate
# ==========================================
print("--- MODEL EVALUATION RESULTS ---")
results = []

for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Calculate Confusion Matrix to extract True Positives, True Negatives, etc.
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()

    # Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)  # Also known as Sensitivity
    specificity = TN / (TN + FP)
    f1 = f1_score(y_test, y_pred)

    # Store results
    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Sensitivity (Recall)": round(recall, 4),
        "Specificity": round(specificity, 4),
        "F1-Score": round(f1, 4)
    })

# ==========================================
# 4. Compare Models
# ==========================================
# Convert results list to a DataFrame for clean formatting
results_df = pd.DataFrame(results).set_index("Model")
print(results_df.to_string())

# Identify the best model based on Accuracy
best_model = results_df['Accuracy'].idxmax()
print(f"\n🏆 Best Performing Model (by Accuracy): {best_model} ({results_df.loc[best_model, 'Accuracy'] * 100:.2f}%)")

# ==========================================
# 5. Visualizing the Comparison
# ==========================================
plt.figure(figsize=(10, 6))
sns.barplot(x=results_df.index, y=results_df['Accuracy'], palette="viridis")
plt.title('Classification Models Comparison - Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Algorithm')
plt.ylim(0.85, 1.0)  # Zooming in on the Y-axis to highlight differences
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
