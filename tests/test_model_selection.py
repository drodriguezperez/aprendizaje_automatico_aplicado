from pytest import approx

import numpy as np
import pandas as pd

from aprendizaje_automatico_aplicado.model_selection import calculate_vif
from aprendizaje_automatico_aplicado.model_selection import select_data_using_vif
from aprendizaje_automatico_aplicado.model_selection import train_validation_test_split


def test_train_validation_test_split():
    """Validación de la función para dividir en tres conjuntos los datos."""

    data_x, data_y = np.arange(200).reshape((100, 2)), range(100)
    x_train, x_val, x_test, y_train, y_val, y_test = train_validation_test_split(
        data_x, data_y)

    assert len(data_x) == 100
    assert len(x_train) == 70
    assert len(x_val) == 15
    assert len(x_test) == 15

    x_train, x_val, x_test, y_train, y_val, y_test = train_validation_test_split(
        data_x, data_y, train_size=0.60, validation_size=0.25, test_size=0.15)

    assert len(data_x) == 100
    assert len(x_train) == 60
    assert len(x_val) == 25
    assert len(x_test) == 15


def test_vif():
    data = pd.DataFrame({'X1': [1, 2, 3, 4, 5], 'X2': [1, 1, 1, 1, 1], 'X3': [
                     2, 2, 3, 4, 4], 'X4': [1, 2, 1, 3, 1]})
                     
    vif = calculate_vif(data)

    assert vif['X1'][0] == approx(10.92592592592594)
    assert vif['X2'][0] == float('inf')
    assert vif['X3'][0] == approx(11.481481481481488)
    assert vif['X4'][0] == approx(1.1851851851851851)

    select = select_data_using_vif(data)

    assert select.shape == (5, 2)

    select = select_data_using_vif(data, max_vif=12)

    assert select.shape == (5, 3)
