import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


def calculateIV(data, features, target):
    result = pd.DataFrame(index=['IV'], columns=features)
    result = result.fillna(0)
    var_target = array(data[target])

    for cat in features:
        var_values = array(data[cat])
        var_levels = unique(var_values)

        mat_values = numpy.zeros(shape=(len(var_levels), 2))

        for i in range(len(var_target)):
            for j in range(len(var_levels)):
                if var_levels[j] == var_values[i]:
                    pos = j
                    break

            # Estimación del número valores en cada nivel
            if var_target[i]:
                mat_values[pos][0] += 1
            else:
                mat_values[pos][1] += 1

            # Obtención del IV
            IV = 0
            for j in range(len(var_levels)):
                if mat_values[j][0] > 0 and mat_values[j][1] > 0:
                    rt = mat_values[j][0] / \
                        (mat_values[j][0] + mat_values[j][1])
                    rf = mat_values[j][1] / \
                        (mat_values[j][0] + mat_values[j][1])
                    IV += (rt - rf) * np.log(rt / rf)

        # Se agrega el IV al listado
        result[cat] = IV

    return result


def calculate_vif(data):
    """Calcula el VIF para un conjunto de datos

    Obtiene el VIF (Factor de inflación de la varianza) de un conjunto de datos.

    Parameters
    ----------
    X : list
        Conjunto de datos

    Returns
    -------
    results : list
        El VIF para cada uno de las caracteristicas de los datos
    """

    features = list(data.columns)
    num_features = len(features)

    model = LinearRegression()

    result = pd.DataFrame(index=['VIF'], columns=features)
    result = result.fillna(0)

    for ite in range(num_features):
        x_features = features[:]
        y_featue = features[ite]
        x_features.remove(y_featue)

        model.fit(data[x_features], data[y_featue])

        if model.score(data[x_features], data[y_featue]) == 1:
            result[y_featue] = float('inf')
        else:
            result[y_featue] = 1 / \
                (1 - model.score(data[x_features], data[y_featue]))

    return result


def evaluate_polynomial(degrees, X, y, true_fun, plot_figure=False):
    """Evaluación de modelos polinómicos
    
    Función que realiza el ajuste a un polinomio de un grado determinado y
    obtiene los valores de modelo mediante validación cruzada.

    Parameters
    ----------
    degrees : int
        Grado del polinomio

    X : list
        Conjunto de datos independientes

    y : list
        Conjunto de datos dependientes

    true_fun : function
        Función original que se desea representar

    plot_figure : boolean
        Variable opcional con la que se indica si se representa la figura con los datos

    Returns
    -------
    scores 
        Resultados del ajuste del modelo
    """

    polynomial_features = PolynomialFeatures(
        degree=degrees, include_bias=False)

    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X[:, np.newaxis], y)

    scores = cross_val_score(
        pipeline, X[:, np.newaxis], y, scoring="neg_mean_squared_error", cv=10)

    if plot_figure:
        X_test = np.linspace(0, 1, 100)
        plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
        plt.plot(X_test, true_fun(X_test), label="True function")
        plt.scatter(X, y, label="Samples")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim((0, 1))
        plt.ylim((-2, 2))
        plt.legend(loc="best")
        plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(degrees, -
                                                           scores.mean(), scores.std()))
    
    return scores


def get_WoE(data, var, target):
    crosstab = pd.crosstab(data[target], data[var])

    print("Obteniendo el Woe para la variable", var, ":")

    for col in crosstab.columns:
        if crosstab[col][1] == 0:
            print("  El WoE para", col,
                  "[", sum(crosstab[col]), "] es infinito")
        else:
            print("  El WoE para", col, "[", sum(crosstab[col]), "] es", np.log(
                float(crosstab[col][0]) / float(crosstab[col][1])))


def plot_dispersion(x, figure_name=None, max_k=10, n_init=10, random_state=1):
    """Representa la distorsion en KMeans

    Ejecuta el algoritmo de KMeans y representando la distorsión frente al número de clústeres.

    Parameters
    ----------
    x : DataFrame
        El conjunot de datos sobre el que se evalua el modelo

    figure_name: String
        Nombre para la figura
    
    max_k: Integer
        Número máximo de clústeres
    
    n_init: Integer
        Número de veces que se ejecuta el agoritmor de KMeans
    
    random_state: Number
        Semilla
    """

    inertia = []

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k, n_init=n_init,
                        random_state=random_state).fit(x)
        inertia.append(kmeans.inertia_)

    plt.plot(range(1, max_k), inertia, 'bx-')
    plt.xlabel('k')
    plt.ylabel(u'Dispersión')
    if figure_name != None:
        plt.title(figure_name)


def plot_sillhouette(x, figure_name=None, max_k=10, n_init=10, random_state=1):
    """Representa la sillhouette en KMeans

    Ejecuta el algoritmo de KMeans y representando la sillhouette frente al número de clústeres.

    Parameters
    ----------
    x : DataFrame
        El conjunot de datos sobre el que se evalua el modelo

    figure_name: String
        Nombre para la figura

    max_k: Integer
        Número máximo de clústeres

    n_init: Integer
        Número de veces que se ejecuta el agoritmor de KMeans

    random_state: Number
        Semilla

    Returns
    -------
    results : Integer
        El k en el que se alcanza el valor máximo de la sillhouette
    """

    sillhouette_avgs = []

    for k in range(2, max_k):
        kmeans = KMeans(n_clusters=k, n_init=n_init,
                        random_state=random_state).fit(x)
        sillhouette_avgs.append(silhouette_score(x, kmeans.labels_))

    plt.plot(range(2, max_k), sillhouette_avgs, 'bx-')
    plt.xlabel('k')
    plt.ylabel(u'Silhouette')
    if figure_name != None:
        plt.title(figure_name)

    return np.argmax(sillhouette_avgs) + 2


def select_data_using_vif(data, max_vif=5):
    result = data.copy(deep=True)

    VIF = calculate_vif(result)

    while VIF.values.max() > max_vif:
        col_max = np.where(VIF == VIF.values.max())[1][0]
        features = list(result.columns)
        features.remove(features[col_max])
        result = result[features]

        VIF = calculate_vif(result)

    return result


def train_validation_test_split(data_x, data_y, train_size=0.70, validation_size=0.15, test_size=0.15, random_state=None):
    """Divide un conjunto de datos en tres: entrenamiento, validación y
       test.
    
    Basándose en el algoritmo train_test_split de Scikit-learn divide un
    conjunot de datos en tres, con diferente peso cada uno.

    Parameters
    ----------
    data_x : list
        Conjunto de datos 

    data_y : list
        Conjunto de datos
    
    train_size : float
        Tamaño de la muestra de entrenamiento

    validation_size : float
        Tamaño de la muestra de validación

    test_size : float
        Tamaño de la muestra de test

    random_state : int, RandomState instance or None, optional (default=None)
        Semilla empleada por el generador de números aleatorios

    Returns
    -------
    results : list
        Lista que contienen los conjuntos de entrenamiento, validación y test.
    """

    validation = validation_size / (test_size + validation_size)

    x_train, x_test, y_train, y_test = train_test_split(
        data_x, data_y, train_size=train_size, random_state=random_state)
    x_val, x_test, y_val, y_test = train_test_split(
        x_test, y_test, train_size=validation, random_state=random_state)

    return x_train, x_val, x_test, y_train, y_val, y_test
