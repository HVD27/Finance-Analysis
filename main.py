# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
df = pd.read_csv("creditcard.csv")

# Troubleshooting: Check for missing values
print(df.isnull().sum())

# Exploratory Data Analysis (EDA)

# Transaction Amount Distribution
plt.figure(figsize=(10,6))
sns.histplot(df['Amount'], bins=50, kde=False)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()

# Transaction Time Analysis
df['Hour'] = df['Time'].apply(lambda x: np.floor(x / 3600) % 24)
plt.figure(figsize=(10,6))
sns.histplot(df['Hour'], bins=24, kde=False)
plt.title('Distribution of Transactions by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Transactions')
plt.show()

# Fraud vs. Non-Fraud Transaction Analysis
plt.figure(figsize=(10,6))
sns.boxplot(x='Class', y='Amount', data=df)
plt.yscale('log')
plt.title('Transaction Amounts by Fraud vs. Non-Fraud')
plt.xlabel('Class (0: Non-Fraud, 1: Fraud)')
plt.ylabel('Transaction Amount (log scale)')
plt.show()

# Correlation Matrix for Fraudulent Transactions
fraud_df = df[df['Class'] == 1]
plt.figure(figsize=(15,10))
sns.heatmap(fraud_df.corr(), cmap='coolwarm', annot=False)
plt.title('Correlation Matrix for Fraudulent Transactions')
plt.show()

# Data Preparation

# Splitting and Scaling Data
X = df.drop(['Class'], axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Customization

# Balancing Classes with SMOTE and Training Balanced Random Forest
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

balanced_rf_classifier = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
balanced_rf_classifier.fit(X_res, y_res)

y_pred_balanced = balanced_rf_classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred_balanced))
print(classification_report(y_test, y_pred_balanced))

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=balanced_rf_classifier, param_grid=param_grid, cv=5, scoring='f1')
grid_search.fit(X_res, y_res)

best_model = grid_search.best_estimator_

y_pred_tuned = best_model.predict(X_test)
print(confusion_matrix(y_test, y_pred_tuned))
print(classification_report(y_test, y_pred_tuned))

# Refined Prediction Algorithms

# Anomaly Detection with Isolation Forest
isolation_forest = IsolationForest(contamination=0.001, random_state=42)
df['Anomaly_Score'] = isolation_forest.fit_predict(df.drop(columns=['Class']))

sns.boxplot(x='Anomaly_Score', y='Amount', data=df)
plt.title('Anomaly Scores by Transaction Amount')
plt.show()

anomalies = df[df['Anomaly_Score'] == -1]
print(anomalies.head())

# Predicting Future Expenses Using ARIMA
df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' is in datetime format
df['Month'] = df['Date'].dt.to_period('M')
monthly_expenses = df.groupby('Month')['Amount'].sum()

model = ARIMA(monthly_expenses, order=(5,1,0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=6)
print(forecast)

monthly_expenses.plot(label='Historical Expenses', figsize=(10,6))
forecast.plot(label='Forecasted Expenses')
plt.legend()
plt.show()

# Insights Generation

# Spending Breakdown by Category
# Assuming we had a 'Category' column
# Uncomment the following lines if you have a 'Category' column in your dataset
# category_spending = df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
# plt.figure(figsize=(10,6))
# sns.barplot(x=category_spending.values, y=category_spending.index)
# plt.title('Spending Breakdown by Category')
# plt.xlabel('Amount Spent (INR)')
# plt.show()

# Identifying High Spending Months and Saving Opportunities
high_spending_months = monthly_expenses[monthly_expenses > monthly_expenses.mean() + 2 * monthly_expenses.std()]
print("High spending months:\n", high_spending_months)

average_spending = monthly_expenses.mean()
suggested_saving = monthly_expenses[monthly_expenses < average_spending].sum()
print(f"Suggested saving opportunity: {suggested_saving} INR")
