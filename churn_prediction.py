# ===============================
# Customer Churn Prediction
# ===============================

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# -------------------------------
# 2. Load dataset
# -------------------------------
df = pd.read_csv("churn_data.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# -------------------------------
# 3. Data cleaning
# -------------------------------

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Remove missing values
df.dropna(inplace=True)

# Drop customerID (not useful for prediction)
df.drop('customerID', axis=1, inplace=True)

print("\nMissing values after cleaning:\n", df.isnull().sum())

# -------------------------------
# 4. Exploratory Data Analysis
# -------------------------------

# Churn distribution
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

# Monthly Charges vs Churn
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

# -------------------------------
# 5. Encode categorical variables
# -------------------------------
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# -------------------------------
# 6. Feature & target split
# -------------------------------
X = df.drop('Churn', axis=1)
y = df['Churn']

# -------------------------------
# 7. Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 8. Feature scaling
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 9. Logistic Regression Model
# -------------------------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# -------------------------------
# 10. Random Forest Model
# -------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_rf))

# -------------------------------
# 11. Feature Importance
# -------------------------------
feature_importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

feature_importance.head(10).plot(kind='bar')
plt.title("Top Factors Influencing Customer Churn")
plt.show()

# -------------------------------
# 12. Final Business Insight
# -------------------------------
print("\nBusiness Insight:")
print("Customers with month-to-month contracts, higher monthly charges,")
print("and shorter tenure are more likely to churn.")
print("Targeted retention strategies can reduce customer loss.")
