import numpy as np
# iris neural network

"""
INFORMATII RETEA NEURONALA

-->
neuroni_intrare = 4  # numarul de caracteristici din setul Iris (sepal length, sepal width, petal length, petal width)
neuroni_strat_ascuns = 10  # numarul de neuroni din stratul ascuns
neuroni_iesire = 3  # numarul de clase din setul Iris (setosa, versicolor, virginica)

nr_ponderi_ascuns = neuroni_intrare * neuroni_strat_ascuns
nr_praguri_ascuns = neuroni_strat_ascuns
nr_ponderi_iesire = neuroni_strat_ascuns * neuroni_iesire
nr_praguri_iesire = neuroni_iesire

dimensiune_cromozom = nr_ponderi_ascuns + nr_praguri_ascuns + nr_ponderi_iesire + nr_praguri_iesire
<--

incapsulez toate informatiile legate de reteaua neuronala in clasa de mai jos.
"""
class ReteaNeuronala:
    def __init__(self, neuroni_intrare: int, neuroni_strat_ascuns: int, neuroni_iesire: int):
        self.neuroni_intrare = neuroni_intrare
        self.neuroni_strat_ascuns = neuroni_strat_ascuns
        self.neuroni_iesire = neuroni_iesire

        self.nr_ponderi_ascuns = self.neuroni_intrare * self.neuroni_strat_ascuns
        self.nr_praguri_ascuns = neuroni_strat_ascuns
        self.nr_ponderi_iesire = neuroni_strat_ascuns * neuroni_iesire
        self.nr_praguri_iesire = neuroni_iesire

        self.dimensiune_cromozom = self.nr_ponderi_ascuns + self.nr_praguri_ascuns + self.nr_ponderi_iesire + self.nr_praguri_iesire
        
    def decodificare_cromozom(self, cromozom):
        ponderi_ascuns = cromozom[:self.nr_ponderi_ascuns].reshape((self.neuroni_intrare, self.neuroni_strat_ascuns))
        
        praguri_ascuns = cromozom[self.nr_ponderi_ascuns:self.nr_ponderi_ascuns + self.nr_praguri_ascuns]

        inceput_iesire = self.nr_ponderi_ascuns + self.nr_praguri_ascuns

        ponderi_iesire = cromozom[inceput_iesire:inceput_iesire + self.nr_ponderi_iesire].reshape((self.neuroni_strat_ascuns, self.neuroni_iesire))
        
        praguri_iesire = cromozom[inceput_iesire + self.nr_ponderi_iesire:inceput_iesire + self.nr_ponderi_iesire + self.nr_praguri_iesire]

        return {
            "ponderi_ascuns": ponderi_ascuns,
            "praguri_ascuns": praguri_ascuns,
            "ponderi_iesire": ponderi_iesire,
            "praguri_iesire": praguri_iesire
        }

     # functia sigmoid
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
        
    # functia de propagare inainte
    def propagare_inainte(self, X, cromozom):
        cromozom_decodificat = self.decodificare_cromozom(cromozom)
        input_ascuns = np.dot(X, cromozom_decodificat['ponderi_ascuns']) + cromozom_decodificat['praguri_ascuns']
        output_ascuns = self.sigmoid(input_ascuns)
        
        input_iesire = np.dot(output_ascuns, cromozom_decodificat['ponderi_iesire']) + cromozom_decodificat['praguri_iesire']
        output_iesire = self.sigmoid(input_iesire)
        
        return output_iesire
    
    
    

    
