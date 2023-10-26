# Importeer de benodigde biblitheken
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor  # Hier vervang ik LinearRegression door DecisionTreeRegressor
import pandas as pd

# Importeer het dataset
data = pd.read_excel('heart.xlsx')

# Hierbinnen definieer je de X en Y variabelen, de "onafhankelijke" en "afhankelijke" variabelen
X = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = data['target']  # Hier vervang ik 'target' door je daadwerkelijke doelvariabele

# Split the data into training and testing sets (if not using cross_val_score)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create your machine learning model (replace DecisionTreeRegressor with your chosen model)
model = DecisionTreeRegressor()  # Hier gebruik ik DecisionTreeRegressor als voorbeeld

# Perform k-fold cross-validation (replace 'cv' with the number of folds you desire)
cv_scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation

# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)

# Calculate and print the mean and standard deviation of the cross-validation scores
mean_cv_score = cv_scores.mean()
std_cv_score = cv_scores.std()
print("Mean CV Score:", mean_cv_score)
print("Standard Deviation of CV Scores:", std_cv_score)
