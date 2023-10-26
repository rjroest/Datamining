# Importeer de benodigde biblitheken
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Importeer het dataset
data = pd.read_excel('heart.xlsx')

# Hierbinnen definieer je de X en Y variabelen, de "onafhankelijke" en "afhankelijke" variabelen
X = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = data['target']  # Hier vervang ik 'target' door je daadwerkelijke doelvariabele

# Maak het KNN-model met het gewenste aantal buren (bijvoorbeeld 5)
model = KNeighborsClassifier(n_neighbors=5)

# Train het model op de gegeven dataset
model.fit(X, y)

# Voorspel een nieuwe datapunt (vervang '...' door de feitelijke kenmerken van het nieuwe datapunt)
nieuwe_data_punt = [[20, 2, 2, 147, 264, 1, 0 ,130, 1, 1.6, 1, 1, 2]]
voorspelling = model.predict(nieuwe_data_punt)
print("Voorspelling:", voorspelling)
