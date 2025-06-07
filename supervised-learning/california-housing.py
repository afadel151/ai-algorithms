import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import mean_squared_error,classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
data = fetch_california_housing(as_frame=True)

df = data.frame

# print(df.info())
# print(df.describe())
# visualise relationships
# sns.pairplot(df, vars=['MedInc','AveRooms','HouseAge','MedHouseVal'])
# plt.show()

#check for missing values 
# print("Missing values :\n", df.isnull().sum())

# Features and Target
X = df[['MedInc','HouseAge','AveRooms']]
y = df['MedHouseVal']

#Split Dataset
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,train_size=0.8,random_state=42)


#Train linear regression 
model = LinearRegression()
model.fit(X_train,y_train)

y_predict = model.predict(X_test)
mse = mean_squared_error(y_test,y_predict)

print('Linear regression MSE:',mse)