import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Lees je dataset in (vervang 'dataset.csv' door de daadwerkelijke naam van je dataset)
data = pd.read_excel('heart.xlsx')

# Definieer de onafhankelijke variabelen (kenmerken) en de afhankelijke variabele (doel)
X = data[['age', 'sex', 'trestbps', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'thal']]  # Voeg hier de namen van je kenmerken toe
y = data['target']  # Voeg hier de naam van je doelvariabele toe

# Lijsten om resultaten op te slaan
k_values = []
accuracy_values = []

# Loop over verschillende 'k'-waarden en bereken de nauwkeurigheid voor elk
for k in range(1, 21):  # Hier loop ik van 1 tot 20, maar je kunt dit aanpassen aan je behoeften
    # Splitst de dataset in trainings- en testgegevens
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=100)

    # Maak een KNN-classificatiemodel met de huidige 'k'-waarde
    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    # Train het model op de trainingsgegevens
    knn_classifier.fit(X_train, y_train)

    # Maak voorspellingen op de testgegevens
    y_pred = knn_classifier.predict(X_test)

    # Bereken de nauwkeurigheid van het model en sla op
    accuracy = accuracy_score(y_test, y_pred)
    k_values.append(k)
    accuracy_values.append(accuracy)

# Maak een lijnplot om de relatie tussen 'k' en nauwkeurigheid te visualiseren
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_values, marker='o', linestyle='-')
plt.title('Nauwkeurigheid vs. Aantal K-Metingen (K-Nearest Neighbors)')
plt.xlabel('Aantal K-Metingen (k)')
plt.ylabel('Nauwkeurigheid')
plt.grid(True)
plt.xticks(range(1, 21))  # Geef x-as labels van 1 tot 20

# Voeg datalabels toe aan de punten op de grafiek
for i, (k, acc) in enumerate(zip(k_values, accuracy_values)):
    plt.text(k, acc, f'{acc:.2f}', fontsize=12, ha='right', va='bottom')

plt.show()

#plot 1 is met test size 0.2 en random state 42
#plot 2 is met test size 0.5 en random state 42
#plot 3 is met test size 0.5 en random size 100 
