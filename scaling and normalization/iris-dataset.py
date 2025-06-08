from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Load iris dataset 

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target


# Dataset info 
# print(X.describe())
# print(data.target_names)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)


y_pred = knn.predict(X_test)

print("Accuracy :\n", accuracy_score(y_test,y_pred))

# Apply min-max scaling 
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train_scaled,X_test_scaled,y_train_scaled,y_test_scaled = train_test_split(X_scaled,y,test_size=0.2,random_state=42)
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled,y_train_scaled)

# predict and evaluate 
y_pred_scaled = knn_scaled.predict(X_test_scaled)

print("Accuracy min-max scaled :",accuracy_score(y_test_scaled,y_pred_scaled))