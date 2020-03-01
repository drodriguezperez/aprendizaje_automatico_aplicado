# Comprobación básica del funcionamiento de las funciones apriori

from pytest import approx

from aprendizaje_automatico_aplicado import apriori

def test_apriori():
    """Comprobación básica del funcionamiento de las funciones apriori"""

    dataset = [['Pan', 'Leche'],
        ['Pan', 'Pañales', 'Cerveza', 'Huevos'],
        ['Leche', 'Pañales', 'Cerveza', 'Cola'],
        ['Leche', 'Pan', 'Pañales', 'Cerveza'],
        ['Pañales', 'Pan', 'Leche', 'Cola'],
        ['Pan', 'Leche', 'Pañales'],
        ['Pan', 'Cola']]
    
    F, soporte = apriori.apriori(dataset, min_support = 0.55, verbose = True)
    
    assert len(F) == 3
    assert len(soporte) == 10