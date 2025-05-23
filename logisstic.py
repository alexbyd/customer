import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
#import pyspark as ps
#from pyspark.sql import SparkSession
#spark=SparkSession.builder.appName('Practise').getOrCreate()
# docker run -it --rm spark:python3 /opt/spark/bin/pyspark
#pip install pyspark

# modelar la incertidumbre, en particular la de naturaleza probabilística

data = pd.read_csv("customer marketing_campaign.csv", sep='\t', engine='python') # El tipo se separa con \t y com es diferente a las (,) de csv toca especificarlo

# Limpieza de datos
# Seleccionamos las columnas numéricas como características
features = ['Year_Birth', 'Kidhome', 'Teenhome', 'Recency', 'MntWines',
            'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
            'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
            'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
            'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4',
            'AcceptedCmp5', 'Complain']

# Variable objetivo
target = 'Response'

data = data.dropna(subset=[target])
X = data[features]
y = data[target]

print(f"Filas originales: {len(data)}")
print("\nForma del Dataset: ", data.shape)

# Identificar columnas numéricas continuas para aplicar IQR
# Excluir variables binarias o categóricas (AcceptedCmp1-5, Complain)
continuous_features = ['Year_Birth', 'Recency', 'MntWines', 'MntFruits',
                      'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                      'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                      'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']

if X[continuous_features].isna().any().any():
 #   Imputando valores nulos en columnas continuas
    X.loc[:, continuous_features] = X[continuous_features].fillna(X[continuous_features].mean())


# Función para eliminar outliers usando IQR
def remove_outliers(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Filtrar filas donde la columna está dentro de los límites
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

# Aplicar IQR a las columnas continuas
data_clean = remove_outliers(data, continuous_features)

# Actualizar X e y después de la limpieza
X = data_clean[features].fillna(data_clean[features].mean())  # Imputar valores nulos restantes
y = data_clean[target]

# Verificar el impacto de la limpieza
print(f"\nFilas originales: {len(data)}")
print(f"Filas después de IQR: {len(data_clean)}")
print(f"Proporción de Response=1 antes: {data[target].mean():.4f}")
print(f"Proporción de Response=1 después: {data_clean[target].mean():.4f}\n")

# Dividimos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarización de datos
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

#MinMaxScaler para normalizar los datos al rango [0,1]

# Convertimos a matrices NumPy
X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()
y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

# Verificamos las dimensiones
print("X_train shape:", X_train_np.shape)
print("X_test shape:", X_test_np.shape)
print("y_train shape:", y_train_np.shape)
print("y_test shape:", y_test_np.shape ,"\n")

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


X_train_np = X_train_norm
y_train_np = y_train.to_numpy()

w, b = train_logistic_regression(X_train_np, y_train_np, learning_rate=0.01, epochs=1000)

# Predicción en conjunto de prueba
z_test = np.dot(X_test_norm, w) + b
y_pred = sigmoid(z_test)
y_pred_class = (y_pred >= 0.5).astype(int)  # Clasificación binaria

# Evaluación
accuracy = accuracy_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)
auc = roc_auc_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")






'''

 git clone https://github.com/openai/spinningup.git
cd spinningup
pip install -e .


def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):

    This function optimizes w and b by running a gradient descent algorithm
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

'''