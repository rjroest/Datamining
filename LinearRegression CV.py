# Importeer de benodigde biblitheken
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Replace with your chosen model
import pandas as pd

# Importeer het dataset
data = pd.read_excel('heart.xlsx')

# Hierbinnen definieeren we de X en Y variabelen. De "dependent" en "independent" variabelen
X = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = data['target']  # Replace with your target variable

# De onderstaande regel split de data in training en testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=100)

# Maak het model van de Linear Regression
model = LinearRegression()

# Perform k-fold cross-validation (replace 'cv' with the number of folds you desire)
cv_scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation

# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)

# Calculate and print the mean and standard deviation of the cross-validation scores
mean_cv_score = cv_scores.mean()
std_cv_score = cv_scores.std()
print("Mean CV Score:", mean_cv_score)
print("Standard Deviation of CV Scores:", std_cv_score)
