import numpy as np
from retea_neuronala import *


# initizalizarea populatiei cu valori aleatoare
def initializarea_populatiei(dimensiune_populatie, dimensiune_cromozom):
    return [np.random.uniform(-1, 1, dimensiune_cromozom) for _ in range(dimensiune_populatie)]

# functia de fitness
def fitness(cromozom, retea, X_train, y_train):
    #PO
    predictii = retea.propagare_inainte(X_train, cromozom)
    #P1
    suma_erori = 0
    #P2
    numar_elem = len(X_train)
    #P3
    for i in range(numar_elem):
        #P4
        eroare = 0
        #P5
        for j in range(len(y_train[i])):
            #P6
            diferenta = y_train[i][j] - predictii[i][j]
            #P7
            eroare += diferenta ** 2
            #P8
        suma_erori += eroare
        #P9
    mse = suma_erori / numar_elem 
    #P10
    return -mse



# selectia parintilor
def selectie_parinti(populatie, fitnessuri, k=2):
    parinti = []
    for i in range(2):
        membri =[]
        for i in range (k):
            indice = np.random.randint(0, len(populatie)) 
            membri.append((indice, fitnessuri[indice])) 
        parinti.append(populatie[ max(membri, key=lambda x: x[1])[0]])
    return parinti

# incrucisarea
def incrucisare(parinte1, parinte2):
    punct = np.random.randint(1, len(parinte1) - 1)
    copil1 = np.concatenate((parinte1[:punct], parinte2[punct:]))
    copil2 = np.concatenate((parinte2[:punct], parinte1[punct:]))
    return copil1, copil2

# apel de functie pentru testarea functionalitatii.
#print(incrucisare(test_cromozomi[0], test_cromozomi[1]))

# mutatia
def mutatie(cromozom, rata_mutatie=0.1):
    for i in range(len(cromozom)):
        if np.random.rand() < rata_mutatie:
            cromozom[i] += np.random.uniform(-0.5, 0.5)
    return cromozom

# algoritmul genetic

def algoritm_genetic(retea, X_train, y_train, dimensiune_populatie=50, generatii=100, rata_mutatie=0.1, prag_convergenta=1e-6):
    fitness_rezultate = []
    populatie = initializarea_populatiei(dimensiune_populatie, retea.dimensiune_cromozom)

    fitness_maxim_anterior = float('-inf')  

    for generatie in range(generatii):
        fitness_rezultate = [fitness(cromozom, retea, X_train, y_train) for cromozom in populatie]

        fitness_maxim_curent = max(fitness_rezultate)
        if abs(fitness_maxim_curent - fitness_maxim_anterior) < prag_convergenta:
            print(f"Algoritmul a convergent la generația {generatie} cu fitness maxim: {fitness_maxim_curent}")
            break

        fitness_maxim_anterior = fitness_maxim_curent
        noua_populatie = []
        for _ in range(dimensiune_populatie // 2):
            parinte1, parinte2 = selectie_parinti(populatie, fitness_rezultate)
            copil1, copil2 = incrucisare(parinte1, parinte2)
            copil1 = mutatie(copil1, rata_mutatie)
            copil2 = mutatie(copil2, rata_mutatie)
            noua_populatie.append(copil1)
            noua_populatie.append(copil2)

        populatie = noua_populatie

    return populatie[np.argmax(fitness_rezultate)]
