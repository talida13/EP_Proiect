from retea_neuronala import ReteaNeuronala
from procesarea_datelor import incarca_si_proceseaza_date
from evolutie import *
import numpy as np

# 1. incarcarea si prelucrarea datelor
X_train, X_test, y_train, y_test = incarca_si_proceseaza_date()


# 2. initializare retea neuronala
retea: ReteaNeuronala = ReteaNeuronala(neuroni_intrare=4, neuroni_strat_ascuns=10, neuroni_iesire=3)

# 3. rularea algoritmului evolutiv
dimensiune_populatie = 50
generatii = 1000
rata_mutatie = 0.05

cel_mai_bun_cromozom= algoritm_genetic(
    retea=retea,
    X_train=X_train,
    y_train=y_train,
    dimensiune_populatie=dimensiune_populatie,
    generatii=generatii,
    rata_mutatie=rata_mutatie
)

# 4. testarea si afisarea rezultatelor
#predictii_train = retea.propagare_inainte(X_train, cel_mai_bun_cromozom)
predictii_test = retea.propagare_inainte(X_test, cel_mai_bun_cromozom)
mse_test = np.mean((y_test - predictii_test) ** 2)

predictii_train = retea.propagare_inainte(X_train, cel_mai_bun_cromozom)
mse_train = np.mean((y_train - predictii_train) ** 2)


# Pentru setul de antrenare
predictii_clase_train = np.argmax(predictii_train, axis=1)
clase_reale_train = np.argmax(y_train, axis=1)
acuratete_train = np.mean(predictii_clase_train == clase_reale_train)

# Pentru setul de test
predictii_clase_test = np.argmax(predictii_test, axis=1)
clase_reale_test = np.argmax(y_test, axis=1)
acuratete_test = np.mean(predictii_clase_test == clase_reale_test)

print("Acuratețea pe setul de antrenare:", acuratete_train)
print("Acuratețea pe setul de test:", acuratete_test)


# Rezultate
print(f"EROARE TRAIN (MSE): {mse_train}")
print(f"EROARE TEST (MSE): {mse_test}")



iris_exemplu = np.array([5.1, 3.5, 1.4, 0.2])  # Iris-setosa

# normalizare
iris_exemplu_normalizat = ((iris_exemplu - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0))).values

predictie = retea.propagare_inainte(iris_exemplu_normalizat.reshape(1, -1), cel_mai_bun_cromozom)

clasa_predusa = np.argmax(predictie)
clase = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
nume_clasa_predusa = clase[clasa_predusa]

print(f"Iris : {iris_exemplu}")
print(f"Predictia: {predictie}")
print(f"Clasa : {nume_clasa_predusa}")





