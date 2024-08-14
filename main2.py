# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv("creditcard.csv")

# Feature Engineering
df['Hour'] = df['Time'].apply(lambda x: np.floor(x / 3600) % 24)
df['Day'] = df['Time'].apply(lambda x: np.floor(x / (3600 * 24)) % 7)
df['Amount_Log'] = np.log1p(df['Amount'])  # Log transformation of the Amount to handle skewness

# Simulate a 'Category' column for financial insights
np.random.seed(42)
categories = ['Groceries', 'Dining', 'Utilities', 'Shopping', 'Transport', 'Miscellaneous']
df['Category'] = np.random.choice(categories, size=len(df), p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])

# Simulate a 'Date' column based on the 'Time' column
start_date = pd.to_datetime('2024-01-01')
df['Date'] = start_date + pd.to_timedelta(df['Time'], unit='s')
df['Month'] = df['Date'].dt.to_period('M')

# Dimensionality Reduction using PCA
pca = PCA(n_components=10)  # Reduce to 10 components for simplicity
df_pca = pca.fit_transform(df.drop(['Class', 'Time', 'Amount', 'Category', 'Date', 'Month'], axis=1))
df_pca = pd.DataFrame(df_pca, columns=[f'PC{i+1}' for i in range(10)])
df = pd.concat([df_pca, df[['Amount_Log', 'Hour', 'Day', 'Category', 'Month', 'Class']]], axis=1)

# One-Hot Encoding for the 'Category' column
df = pd.get_dummies(df, columns=['Category'], drop_first=True)

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(10,6))
sns.histplot(df['Amount_Log'], bins=50, kde=False)
plt.title('Distribution of Log-Transformed Transaction Amounts')
plt.xlabel('Log Amount')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(df['Hour'], bins=24, kde=False)
plt.title('Distribution of Transactions by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Transactions')
plt.show()

# Correlation Matrix for PCA Components
# Exclude non-numeric columns before calculating the correlation matrix
numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns
plt.figure(figsize=(15,10))
sns.heatmap(numeric_df.corr(), cmap='coolwarm', annot=False)
plt.title('Correlation Matrix for PCA Components')
plt.show()

# Data Preparation

# Splitting and Scaling Data
X = df.drop(['Class', 'Month'], axis=1)  # Exclude 'Month' as it's non-numeric
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Customization

# Balancing Classes with SMOTE and Training Balanced Random Forest
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# Explicitly set the parameters to match the future behavior
balanced_rf_classifier = BalancedRandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    sampling_strategy='all',  # Future behavior
    replacement=True,         # Future behavior
    bootstrap=False           # Future behavior
)
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

# Advanced Model Evaluation: ROC-AUC
y_prob = best_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(10,6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Anomaly Detection with Isolation Forest
isolation_forest = IsolationForest(contamination=0.001, random_state=42)
df['Anomaly_Score'] = isolation_forest.fit_predict(df.drop(columns=['Class']))

sns.boxplot(x='Anomaly_Score', y='Amount_Log', data=df)
plt.title('Anomaly Scores by Log-Transformed Transaction Amount')
plt.show()

anomalies = df[df['Anomaly_Score'] == -1]
print(anomalies.head())

# Ensemble Learning: Voting Classifier
gb_classifier = GradientBoostingClassifier(random_state=42)
voting_classifier = VotingClassifier(estimators=[
    ('rf', best_model),
    ('gb', gb_classifier)
], voting='soft')

voting_classifier.fit(X_res, y_res)
y_pred_ensemble = voting_classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred_ensemble))
print(classification_report(y_test, y_pred_ensemble))

# Predicting Future Expenses Using ARIMA
monthly_expenses = df.groupby('Month')['Amount_Log'].sum()

model = ARIMA(monthly_expenses, order=(5,1,0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=6)
print(forecast)

monthly_expenses.plot(label='Historical Expenses', figsize=(10,6))
forecast.plot(label='Forecasted Expenses')
plt.legend()
plt.show()

# Insights Generation

# Monthly Spending Breakdown by Category
monthly_spending_by_category = df.groupby(['Month', 'Category'])['Amount_Log'].sum().unstack().fillna(0)
monthly_spending_by_category.plot(kind='bar', stacked=True, figsize=(12,8))
plt.title('Monthly Spending Breakdown by Category')
plt.xlabel('Month')
plt.ylabel('Log of Amount Spent')
plt.show()

# Identifying Savings Opportunities
average_monthly_spending = monthly_expenses.mean()
savings_opportunities = monthly_expenses[monthly_expenses < average_monthly_spending]
potential_savings = average_monthly_spending - savings_opportunities

print("Potential Savings for Months Below Average Spending:")
print(potential_savings)

plt.figure(figsize=(10,6))
plt.plot(monthly_expenses.index, monthly_expenses, label='Monthly Expenses')
plt.plot(savings_opportunities.index, savings_opportunities, 'ro', label='Savings Opportunities')
plt.axhline(y=average_monthly_spending, color='g', linestyle='--', label='Average Monthly Spending')
plt.title('Monthly Expenses with Savings Opportunities')
plt.xlabel('Month')
plt.ylabel('Log of Amount Spent')
plt.legend()
plt.show()
