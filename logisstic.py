import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import copy



data = pd.read_csv("customer marketing_campaign.csv")
#  obtener el tamano del conjunto de datos, visualizar algunos de sus componentes,
data.shape




# centrar y estandarizar el conjunto de datos. Esto significa que se resta la media
# de todo el array numpy de cada ejemplo y luego se divide cada ejemplo por la
#  desviación estándar del array numpy.
#. https://www.kaggle.com/code/marcinrutecki/standardize-or-normalize-ultimate-answer





# - Determinar las dimensiones y formas del problema (m_train, m_test, num_px, ...)
#  aplanarlos para que todos tengan la misma forma con np.reshape().T,
# - Remodelar los conjuntos de datos para que cada ejemplo sea ahora un vector de tamaño (num_px \* num_px \* 3, 1)





# sigmoid, computar el costo de la funcion

def sigmoid(s):
    return s

# inicializar parametros en cero porque

w,b = 0


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
