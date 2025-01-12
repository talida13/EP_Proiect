import unittest
from retea_neuronala import (
    ReteaNeuronala
)
from evolutie import (
    initializarea_populatiei, selectie_parinti, incrucisare, mutatie, fitness,
    algoritm_genetic
)
from procesarea_datelor import (
    incarca_si_proceseaza_date
)

import numpy as np


class TesteReteaNeuronala(unittest.TestCase):
    def test_initializare_populatie(self):
        populatie_test = initializarea_populatiei(10, 5)
        self.assertEqual(len(populatie_test), 10)
        self.assertEqual(len(populatie_test[0]), 5)

    def test_dimensiune_cromozom(self):
        # calculam manual dimensiunea unui cromozom
        neuroni_intrare = 4
        neuroni_strat_ascuns = 10
        neuroni_iesire = 3

        nr_ponderi_ascuns = neuroni_intrare * neuroni_strat_ascuns
        nr_praguri_ascuns = neuroni_strat_ascuns
        nr_ponderi_iesire = neuroni_strat_ascuns * neuroni_iesire
        nr_praguri_iesire = neuroni_iesire
        dimensiune_cromozom = nr_ponderi_ascuns + nr_praguri_ascuns + nr_ponderi_iesire + nr_praguri_iesire

        # initializam o retea neuronala
        retea = ReteaNeuronala(neuroni_intrare=neuroni_intrare, neuroni_strat_ascuns=neuroni_strat_ascuns, neuroni_iesire=neuroni_iesire)
        
        # comparam dimensiunea cromozomului calculata de noi cu cea calculata de retea.
        self.assertEqual(retea.dimensiune_cromozom, dimensiune_cromozom)

    def test_selectie_parinti(self):
        populatie = [np.random.uniform(-1, 1, 5) for _ in range(10)]
        fitnessuri = np.random.uniform(0, 1, 10)
        parinti = selectie_parinti(populatie, fitnessuri)
        self.assertEqual(len(parinti), 2)

    def test_decodificare_cromozom(self):
        # calculam manual dimensiunea unui cromozom
        neuroni_intrare = 4
        neuroni_strat_ascuns = 10
        neuroni_iesire = 3

        nr_ponderi_ascuns = neuroni_intrare * neuroni_strat_ascuns
        nr_praguri_ascuns = neuroni_strat_ascuns
        nr_ponderi_iesire = neuroni_strat_ascuns * neuroni_iesire
        nr_praguri_iesire = neuroni_iesire
        dimensiune_cromozom = nr_ponderi_ascuns + nr_praguri_ascuns + nr_ponderi_iesire + nr_praguri_iesire

        # cromozom
        cromozom = initializarea_populatiei(1, dimensiune_cromozom=dimensiune_cromozom)[0]

        # retea neuronala
        retea = ReteaNeuronala(neuroni_intrare=neuroni_intrare, neuroni_strat_ascuns=neuroni_strat_ascuns, neuroni_iesire=neuroni_iesire)

        cromozom_decodificat = retea.decodificare_cromozom(cromozom=cromozom)
        
        # verificam valorile obtinute 
        self.assertEqual(cromozom_decodificat["ponderi_ascuns"].shape, (neuroni_intrare, neuroni_strat_ascuns))
        self.assertEqual(len(cromozom_decodificat["praguri_ascuns"]), neuroni_strat_ascuns)
        self.assertEqual(cromozom_decodificat['ponderi_iesire'].shape, (neuroni_strat_ascuns, neuroni_iesire))
        self.assertEqual(len(cromozom_decodificat['praguri_iesire']), neuroni_iesire)
    
    def test_incrucisare(self):
        # initializam o retea
        retea = ReteaNeuronala(neuroni_intrare=4, neuroni_strat_ascuns=10, neuroni_iesire=3)

        # generam doi parinti la intamplare
        parinte1 = np.random.rand(retea.dimensiune_cromozom)
        parinte2 = np.random.rand(retea.dimensiune_cromozom)
        
        # aplicam functia de incrucisare
        copil1, copil2 = incrucisare(parinte1, parinte2)

        # verificam daca 
        self.assertEqual(len(copil1), len(parinte1))
        self.assertEqual(len(copil2), len(parinte2))

    def test_mutatie_100(self):
        retea = ReteaNeuronala(4, 10, 3)
        cromozom = initializarea_populatiei(1, retea.dimensiune_cromozom)[0]

        # functia de mutatie modifica direct argumentul trimis ca parametru asa ca voi face o copie.
        copie = cromozom.copy()

        # testam cu sansa de 100% de mutatie
        rata_mutatie = 1
        cromozom_mutat = mutatie(copie, rata_mutatie)
        
        self.assertFalse(np.array_equal(cromozom, cromozom_mutat))

    def test_mutatie_0(self):
        retea = ReteaNeuronala(4, 10, 3)
        cromozom = initializarea_populatiei(1, retea.dimensiune_cromozom)[0]

        # functia de mutatie modifica direct argumentul trimis ca parametru asa ca voi face o copie.
        copie = cromozom.copy()

        # testam cu sansa de 0% de mutatie
        rata_mutatie = 0
        cromozom_mutat = mutatie(copie, rata_mutatie)
        
        self.assertTrue(np.array_equal(cromozom, cromozom_mutat))

    def test_fitness(self):
        retea = ReteaNeuronala(4, 10, 3)
        cromozom = initializarea_populatiei(1, retea.dimensiune_cromozom)[0]
        X_train, _, Y_train, _ = incarca_si_proceseaza_date()
        rezultat = fitness(cromozom, retea, X_train, Y_train)
        self.assertIsInstance(rezultat, float)

    def test_algoritm_genetic(self):
        retea = ReteaNeuronala(4, 10, 3)
        X_train, _, Y_train, _ = incarca_si_proceseaza_date()
        rezultat = algoritm_genetic(
            retea=retea, X_train=X_train, y_train=Y_train,
            dimensiune_populatie=10, generatii=5, rata_mutatie=0.1
        )
        self.assertEqual(len(rezultat), retea.dimensiune_cromozom)
    
    def test_sigmoid(self):
        x = np.array([-1e10, -1, 0, 1, 1e10])
        retea = ReteaNeuronala(4, 10, 3)
        rezultat = retea.sigmoid(x)
        asteptat = np.array([0, 0.26894142, 0.5, 0.73105858, 1])

        np.testing.assert_almost_equal(rezultat, asteptat, decimal=6)


if __name__ == "__main__":
    unittest.main()