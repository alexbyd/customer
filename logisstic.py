import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from debugpy.common.timestamp import reset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import copy
import scipy
from scipy import constants
import statsmodels.api as sm
# docker run -it --rm spark:python3 /opt/spark/bin/pyspark
#pip install pyspark

# modelar la incertidumbre, en particular la de naturaleza probabilística

data = pd.read_csv("customer marketing_campaign.csv", sep='\t', engine='python') # El tipo se separa con \t y com es diferente a las (,) de csv toca especificarlo

#  obtener el tamano del conjunto de datos, visualizar algunos de sus componentes,


print("\n ---------------------- Columnas ------------------------- \n")

print("columns data from csv \n"+str(data.head()))

print("\n ---------------------- Tipos de datos ------------------------- \n")
print(data.dtypes)
print("\n forma de la matriz \n"+str(data.shape))

'''
for column in data.columns:
    print("\n "+column+"  \n")
    print(data[column].dtype)
'''

print("\n ---------------------- Estadisticos ------------------------- \n")
print(data.describe())

print("\n ---------------------- Valores nulos ------------------------- \n")
print(data.isnull().sum())

print("\n ---------------------- Valores duplicados ------------------------- \n")
print(data.duplicated().sum())

print("\n ---------------------- Valores unicos ------------------------- \n")
print(data.nunique())

#  y obtener un resumen de los datos estadisticos, media mediana moda, varianza,
#  recorrido intercuartilico, coeficiente de desviacion
# medidas de asimetria
#  desviacion standar,

#  correlacion, es la medida mas importante
#
#
#  ver si se puede realizar una tabla de frecuencias,
#  que tipo de distribucion tinene los datos normal, ... hallar la forontera ideal de las distribucione
# las colunas son estadisticamente independientes?
#visualizar con matplotlib o sns cada aspecto estadistico





#  -------------------------------- Normalizacion y estandarizacion   -------------------------


# https://www.datacamp.com/es/tutorial/understanding-logistic-regression-python
# - Determinar las dimensiones y formas del problema (m_train, m_test, num_px, ...)
#  aplanarlos para que todos tengan la misma forma con np.reshape().T,
# - Remodelar los conjuntos de datos para que cada ejemplo sea ahora un vector de tamaño (num_px \* num_px \* 3, 1)
# centrar y estandarizar el conjunto de datos. Esto significa que se resta la media


# Supongamos que tu DataFrame se llama 'df'
# Seleccionamos las columnas numéricas como características
features = ['Year_Birth', 'Kidhome', 'Teenhome', 'Recency', 'MntWines',
            'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
            'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
            'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
            'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4',
            'AcceptedCmp5', 'Complain']

# Variable objetivo
target = 'Income'

# Extraemos X (características) y y (objetivo)
X = data[features]
y = data[target]

# Manejo de valores nulos (si los hay)
X = X.fillna(X.mean())  # Rellenar con la media de cada columna
y = y.fillna(y.mean())  # Rellenar con la media de Income

# Dividimos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertimos a matrices NumPy
X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()
y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

# Verificamos las dimensiones
print("X_train shape:", X_train_np.shape)
print("X_test shape:", X_test_np.shape)
print("y_train shape:", y_train_np.shape)
print("y_test shape:", y_test_np.shape)



scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)  # Ajusta y transforma el conjunto de entrenamiento
X_test_norm = scaler.transform(X_test)        # Solo transforma el conjunto de prueba

# validamos la correlacion de la variable response con las demas
# Correlación entre características y target (si target es Response)
correlation = data[features + ['Response']].corr()
print(correlation['Response'].sort_values(ascending=False))

# --------------- Entrenamiento modelo----------------------------------------------------


def sigmoid(z):
    return 1 / (1 + np.exp(-z))



def cost_function(X, y, w, b):
    m = X.shape[0]
    z = np.dot(X, w) + b
    y_hat = sigmoid(z)
    cost = -np.mean(y * np.log(y_hat + 1e-10) + (1 - y) * np.log(1 - y_hat + 1e-10))
    return cost



def train_logistic_regression(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    w = np.zeros(n)  # Inicializa pesos en 0
    b = 0  # Inicializa sesgo en 0

    for _ in range(epochs):
        # Forward propagation
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)

        # Backward propagation
        dw = (1 / m) * np.dot(X.T, (y_hat - y))  # Gradiente de w
        db = (1 / m) * np.sum(y_hat - y)  # Gradiente de b

        # Actualización de parámetros
        w -= learning_rate * dw
        b -= learning_rate * db


        if _ % 100 == 0:
            cost = cost_function(X, y, w, b)
            print(f"Epoch {_}, Cost: {cost}")

    return w, b


# Aplicamos al conjunto normalizado (asumiendo target = 'Response')
X_train_np = X_train_norm
y_train_np = y_train.to_numpy()
w, b = train_logistic_regression(X_train_np, y_train_np)


# Entrenamiento
w, b = train_logistic_regression(X_train_np, y_train_np, learning_rate=0.01, epochs=1000)

# Predicción en conjunto de prueba
z_test = np.dot(X_test_norm, w) + b
y_pred = sigmoid(z_test)
y_pred_class = (y_pred >= 0.5).astype(int)  # Clasificación binaria

# Evaluación simple
accuracy = np.mean(y_pred_class == y_test)
print(f"Accuracy: {accuracy}")





'''
def standardize(X):
    return (X - np.mean(X)) / np.std(X)
    esto lo que hace esque halla la correlacions estadistica de una variable con otra
    osea cual es la relacion de una columna con otra 

 git clone https://github.com/openai/spinningup.git
cd spinningup
pip install -e .

def propagate(w, b, X, Y): # este codig lo escribio codeium asi que toca revisarlo bien
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    grads = {"dw": dw,
             "db": db}
    return grads, cost

# sigmoid, computar el costo de la funcion

def sigmoid(s):
    return s

# inicializar parametros en cero porque

w,b = np.nan


def initialize_with_zeros(dim):
    w + np.zeros((dim, 1))
    b + 0.
    return w,b


# propagacion hacia adelante y atras
# Softmax



# usar gradient decenden para optimizar parametros

# GRADED FUNCTION: optimize

def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps


------------------------------------------------------------------------------------------------------------------------
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function


######## costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

------------------------------------------------------------------------------------------------------------------------


    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    for i in range(num_iterations):
        # (≈ 1 lines of code)
        # Cost and gradient calculation
        # grads, cost = ...
        # YOUR CODE STARTS HERE

        grads, cost = propagate(w, b, X, Y)

        # YOUR CODE ENDS HERE

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule (≈ 2 lines of code)
        # w = ...
        # b = ...
        # YOUR CODE STARTS HERE

        w = w - learning_rate * dw
        b = b - learning_rate * db

        # YOUR CODE ENDS HERE

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

            # Print the cost every 100 training iterations
            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs
'''