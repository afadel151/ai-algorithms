import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv('./Telco-Customer-Churn.csv')

# Drop customerID since it's just an identifier
df.drop(columns=['customerID'], inplace=True)

# Encode target
le = LabelEncoder()
df['Churn'] = le.fit_transform(df['Churn'])

# Convert all categorical variables to numerical using get_dummies
df = pd.get_dummies(df)

# Split features and target
X = df.drop(columns=['Churn'])
y = df['Churn']

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)

# Train k-NN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Evaluate models
log_pred = log_model.predict(X_test)
kn_pred = knn_model.predict(X_test)

print("Logistic Regression Classification Report:")
print(classification_report(y_test, log_pred))
print("Confusion Matrix for Logistic Regression:")
print(confusion_matrix(y_test, log_pred))

print("\nKNN Classification Report:")
print(classification_report(y_test, kn_pred))
print("Confusion Matrix for KNN:")
print(confusion_matrix(y_test, kn_pred))
