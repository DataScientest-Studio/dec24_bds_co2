import joblib
model = joblib.load("etude2/KNN_pas_optimise.pkl")
print(type(model))  # Vérifier si c'est un KNeighborsClassifier
