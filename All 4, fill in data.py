# Import the necessary libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
import pandas as pd

# Import the dataset
data = pd.read_excel('heart.xlsx')

# Define the X and Y variables, the "independent" and "dependent" variables
X = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = data['target']

# Create the Gaussian Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X, y)

# Create the K-Nearest Neighbors model with the desired number of neighbors (e.g., 5)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X, y)

# Create the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X, y)

# Create the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X, y)  # Fit the model for linear regression

# Define a threshold for classification (e.g., 0.5)
threshold = 0.5

# Predict a new data point (replace '...' with the actual features of the new data point)
new_data_point = [[16, 1, 2, 70, 150, 0, 0, 200, 0, 3.7, 1, 2, 1]]

# Make predictions for each model
nb_prediction = nb_model.predict(new_data_point)
knn_prediction = knn_model.predict(new_data_point)
dt_prediction = dt_model.predict(new_data_point)
lr_prediction = [1] if lr_model.predict(new_data_point) > threshold else [0]

# Print the predictions for each model
print("Gaussian Naive Bayes prediction:", nb_prediction)
print("K-Nearest Neighbors prediction:", knn_prediction)
print("Decision Tree prediction:", dt_prediction)
print("Linear Regression prediction:", lr_prediction)
