# Importeer de benodigde bibliotheken
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # Importeer KNeighborsClassifier voor KNN
import pandas as pd

# Importeer het dataset
data = pd.read_excel('heart.xlsx')

# Definieer de onafhankelijke variabelen (X) en de afhankelijke variabele (y)
X = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = data['target']  # Hier vervang ik 'target' door je daadwerkelijke doelvariabele

# Splits de dataset in trainings- en testsets (als je geen cross_val_score gebruikt)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Maak je machine learning model (vervang KNeighborsClassifier door het gewenste model)
model = KNeighborsClassifier(n_neighbors=5)  # Hier gebruik ik KNeighborsClassifier met 5 buren als voorbeeld

# Voer k-fold cross-validatie uit (vervang 'cv' door het gewenste aantal vouwen)
cv_scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validatie

# Print de cross-validatie scores
print("Cross-Validatie Scores:", cv_scores)

# Bereken en print het gemiddelde en de standaardafwijking van de cross-validatie scores
gemiddelde_cv_score = cv_scores.mean()
standaardafwijking_cv_score = cv_scores.std()
print("Gemiddelde CV Score:", gemiddelde_cv_score)
print("Standaardafwijking van CV Scores:", standaardafwijking_cv_score)
