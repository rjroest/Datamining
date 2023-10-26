# Importeer de benodigde biblitheken
from sklearn.naive_bayes import GaussianNB
import pandas as pd

# Importeer het dataset
data = pd.read_excel('heart.xlsx')

# Hierbinnen definieer je de X en Y variabelen, de "onafhankelijke" en "afhankelijke" variabelen
X = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = data['target']  # Hier vervang ik 'target' door je daadwerkelijke doelvariabele

# Maak het Gaussian Naive Bayes-model
model = GaussianNB()

# Train het model op de gegeven dataset
model.fit(X, y)

# Nu kun je het model gebruiken om voorspellingen te doen op nieuwe gegevens
# Bijvoorbeeld, om een enkele voorspelling te doen:
nieuwe_data_punt = [[20, 2, 2, 147, 264, 1, 0 ,130, 1, 1.6, 1, 1, 2]]  # Vervang '...' door de feitelijke kenmerken van het nieuwe datapunt
voorspelling = model.predict(nieuwe_data_punt)
print("Voorspelling:", voorspelling)
