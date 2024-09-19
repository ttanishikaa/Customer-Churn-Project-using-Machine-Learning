import pandas as pd
data = pd.read_csv('customer_data.csv')
print(data.describe())
data.hist(figsize=(10, 10))
data.fillna(data.mean(), inplace=True)
data = pd.get_dummies(data, columns=['subscription_type'], drop_first=True)
from sklearn.model_selection import train_test_split
X = data.drop('churn', axis=1)
y = data['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
