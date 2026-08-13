import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ==========================================
# PART A: Data Loading & Exploration
# ==========================================
print("--- PART A: LOADING & EXPLORATION ---")

# 1. Load the dataset
# FIXED: Added 'r' before the string to handle Windows backslashes properly
df = pd.read_excel(r"C:\Users\pragy\Downloads\lab1 data.xlsx")

# Rename columns for easier access
column_mapping = {
    'Job role that you are interested in': 'Job_Role',
    'What is the minimum salary of students placed through campus (In LPA..respond as a number)': 'Min_Salary_LPA',
    'What is the maximum salary of students placed through campus (In LPA..respond as a number)': 'Max_Salary_LPA',
    'What is the  median salary of students placed through campus (In LPA..respond as a number)': 'Median_Salary_LPA',
    'Which is the highest paying company': 'Highest_Paying_Company',
    'Rate your contribution towards extra curricular activities': 'Extracurricular_Rating',
    'Rate your technical competencies': 'Technical_Rating',
    'What are your package expectations (LPA)': 'Expected_Salary_LPA',
    'your CIA % of last semester': 'CIA_Percent',
    'your GPA of last semester': 'GPA',
    'Your maximum attendance % till last semester': 'Attendance_Percent',
    'Internships Interests': 'Internship_Interest'
}
df = df.rename(columns=column_mapping)

# 2. Display Exploration Data
print("\n--- First 10 Rows ---")
print(df.head(10))
print("\n--- Last 10 Rows ---")
print(df.tail(10))
print("\n--- Dataset Info ---")
print(df.info())
print("\n--- Summary Statistics ---")
print(df.describe(include='all'))

# Save missing values for visualization later
missing_before = df.isna().sum()

# ==========================================
# PART B: Data Quality Assessment
# ==========================================
print("\n--- PART B: QUALITY ASSESSMENT ---")
print("\n1. Missing Values per Column:\n", missing_before)
print("\n2. Duplicate Records (based on Reg No):", df.duplicated(subset=['Reg No']).sum())
print("\n3. Unique Values in Expected Salary:\n",
      df['Expected_Salary_LPA'].unique()[:10])  # Limiting to 10 for terminal readability
print("\n4. Unique Values in Job Role:\n", df['Job_Role'].unique())
print("\n4. Unique Values in Internship Interest:\n", df['Internship_Interest'].unique())

# ==========================================
# PART C: Data Cleaning Tasks
# ==========================================
print("\n--- PART C: DATA CLEANING ---")


# 1. Standardize Salary Columns
def clean_salary(val):
    if pd.isna(val):
        return np.nan
    val = str(val).upper().strip()
    # If standardizing '1Cr' to 100 LPA
    if 'CR' in val:
        num = re.sub(r'[^\d\.]', '', val)
        return float(num) * 100 if num else np.nan
    # Extract just the numbers for LPA
    num = re.sub(r'[^\d\.]', '', val)
    return float(num) if num else np.nan


salary_cols = ['Min_Salary_LPA', 'Max_Salary_LPA', 'Median_Salary_LPA', 'Expected_Salary_LPA']
for col in salary_cols:
    df[col] = df[col].apply(clean_salary)


# 2. Standardize Percentage Columns
def clean_percentage(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    val = val.replace('%', '')  # Remove % symbol
    try:
        num = float(val)
        # If the number is a decimal less than or equal to 1 (e.g. 0.77), convert to 77
        if num <= 1.0:
            num = num * 100
        return num
    except ValueError:
        return np.nan


percent_cols = ['CIA_Percent', 'Attendance_Percent']
for col in percent_cols:
    df[col] = df[col].apply(clean_percentage)

# Ensure GPA is numeric as well
df['GPA'] = pd.to_numeric(df['GPA'].astype(str).str.replace(r'[^\d\.]', '', regex=True), errors='coerce')

# 3. Fix Categorical Data
df['Job_Role'] = df['Job_Role'].str.title().str.strip()
df['Job_Role'] = df['Job_Role'].replace({'Data Sci': 'Data Scientist', 'Ai Engineer': 'AI Engineer'})
df['Internship_Interest'] = df['Internship_Interest'].str.title().str.strip()

# 4. Handle Timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# 5. Remove Duplicate Records based on Reg No
df = df.drop_duplicates(subset=['Reg No'], keep='first')
print(f"Dataset shape after removing duplicates: {df.shape}")

# ==========================================
# PART D: Handling Missing Values
# ==========================================
print("\n--- PART D: HANDLING MISSING VALUES ---")

# 1. Drop rows with very high missing values (e.g., > 40% missing)
threshold = len(df.columns) * 0.60  # Require at least 60% non-NA values
df = df.dropna(thresh=threshold)

# 2. Imputation for numeric columns (Median)
numeric_impute_cols = ['GPA', 'Attendance_Percent', 'Extracurricular_Rating', 'Technical_Rating'] + salary_cols
for col in numeric_impute_cols:
    df[col] = df[col].fillna(df[col].median())

# 3. Imputation for categorical columns (Mode)
categorical_impute_cols = ['Job_Role', 'Internship_Interest', 'Highest_Paying_Company']
for col in categorical_impute_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Save missing values after cleaning
missing_after = df.isna().sum()
print("Missing values after imputation:\n", missing_after)

# ==========================================
# PART E: Outlier Detection and Treatment
# ==========================================
print("\n--- PART E: OUTLIER TREATMENT ---")


def cap_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Capping the outliers
    dataframe[column] = np.where(dataframe[column] < lower_bound, lower_bound, dataframe[column])
    dataframe[column] = np.where(dataframe[column] > upper_bound, upper_bound, dataframe[column])
    return dataframe


outlier_cols = ['Expected_Salary_LPA', 'Attendance_Percent', 'GPA']
for col in outlier_cols:
    df = cap_outliers_iqr(df, col)

print("Outliers capped successfully using IQR method.")

# ==========================================
# PART F: Visualization
# ==========================================
print("\n--- PART F: VISUALIZATIONS ---")

sns.set_theme(style="whitegrid")

# 1. Missing Values Before and After Cleaning
fig, ax = plt.subplots(figsize=(12, 6))
width = 0.35
x = np.arange(len(missing_before))

ax.bar(x - width / 2, missing_before, width, label='Before Cleaning', color='salmon')
ax.bar(x + width / 2, missing_after, width, label='After Cleaning', color='lightgreen')

ax.set_ylabel('Number of Missing Values')
ax.set_title('Missing Values per Column: Before vs After Cleaning')
ax.set_xticks(x)
ax.set_xticklabels(missing_before.index, rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.show()

# 2. Boxplots for Outlier Detection (Post-Capping)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.boxplot(y=df['Expected_Salary_LPA'], ax=axes[0], color='skyblue')
axes[0].set_title('Expected Salary (LPA)')

sns.boxplot(y=df['Attendance_Percent'], ax=axes[1], color='lightgreen')
axes[1].set_title('Attendance %')

sns.boxplot(y=df['GPA'], ax=axes[2], color='lightcoral')
axes[2].set_title('GPA')

plt.suptitle('Boxplots for Key Metrics (Outliers Capped)')
plt.tight_layout()
plt.show()

print("\nData cleaning and processing completed successfully!")