# Introducción

El gasto en publicidad puede afectar drasticamente el rendimiento de los negocios,
y la implementacion de modelos estadisticos de predicción es esencial para 
tomar decisiones informadas sobre la asignacion de recursos.

## Objetivos

El objetivo de este proyecto es predecir quien respondera a una oferta de un 
producto o servicio.

La principal variable a predecir es la cantidad de personas que responderan a la oferta.
Los buenos resultados traeran consigo mayores ingresos, además buscamos con el modelo estadistico mejorar los 
procedimientos operativos en las campañas de publicidad.



## Resumen de datos

**Fuentes de datos**

Kaggle  **[Marketing Campaign](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign)**


 **Descripción de los datos**

Tipo de variables:

Education, Marital_Status, Dt_Customer = object

Income = float64

Las demás variables son de tipo int64


- Respuesta (objetivo): 1 si el cliente aceptó la oferta en la última campaña, 0 en caso contrario.

- Queja: 1 si el cliente se quejó en los últimos 2 años.

- DtCustomer: fecha de alta del cliente en la empresa.

- Education: nivel de educación del cliente.

- Marital: estado civil del cliente.

- Kidhome: número de niños pequeños en el hogar del cliente.

- Teenhome: número de adolescentes en el hogar del cliente.

- Income: ingresos anuales del hogar del cliente. Ingresos

- MntFishProducts: cantidad gastada en productos pesqueros en los últimos 2 años

- MntMeatProducts: cantidad gastada en productos cárnicos en los últimos 2 años

- MntFruits: cantidad gastada en productos de fruta en los últimos 2 años

- MntSweetProducts: cantidad gastada en productos dulces en los últimos 2 años

- MntWines: cantidad gastada en productos de vino en los últimos 2 años

- MntGoldProds: cantidad gastada en productos de oro en los últimos 2 años

- NumDealsPurchases: número de compras realizadas con descuento

- NumCatalogPurchases: número de compras realizadas por catálogo

- NumStorePurchases: número de compras realizadas directamente en tiendas físicas

- NumWebPurchases: número de compras realizadas a través del sitio web de la empresa

- NumWebVisitsMonth: número de visitas al sitio web de la empresa en el último mes

- Recentency: número de días desde la última compra

- AcceptedCmp1: 1 si el cliente aceptó la oferta en la primera campaña, 0 en caso contrario.
- AcceptedCmp2: 1 si el cliente aceptó la oferta en la segunda campaña, 0 en caso contrario.
- AcceptedCmp3: 1 si el cliente aceptó la oferta en la tercera campaña, 0 en caso contrario.
- AcceptedCmp4: 1 si el cliente aceptó la oferta en la cuarta campaña, 0 en caso contrario.
- AcceptedCmp5: 1 si el cliente aceptó la oferta en la quinta campaña, 0 en caso contrario.



**Limpieza de datos y suposiciones**
    
 
   Corrección de valores atípicos: Detectar y tratar outliers 
   
Corrección de valores faltantes: Imputación de datos faltantes

#  Metodología



**Regresión logística**

Problemas de clasificación binaria. Modela la probabilidad de que una
observación pertenezca a una clase
usando la función logística:
    
$P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots)}}$


Predice probabilidades y asigna clases (0, 1).

**Uso en el contexto**

Queremos predecir si un cliente responderá positivamente a una oferta de un producto o servicio.
Esto ayuda a la planeación estratégica: identificar productos o servicios con mayor potencial de ventas y permite priorizar recursos.

**Por qué se elige**

Clasificación binaria: El problema es binario (Respuesta positiva o negativa).

Probabilidades: Proporciona probabilidades, útiles para decisiones basadas en riesgo (por ejemplo, "80% de probabilidad de aceptar la oferta").

<!--
## Interpretación
Predicciones: El modelo predice correctamente que X_test[0] (ventas=150) es clase 0 (ventas bajas) y X_test[1] (ventas=250) es clase 1 (ventas altas).

Probabilidades: Muestra la probabilidad de cada clase (por ejemplo, 75% de probabilidad de ventas altas para X_test[1]).

Coeficientes: Un coeficiente positivo para ventas indica que un aumento en ventas incrementa la probabilidad de clase 1; un coeficiente negativo para precio sugiere que precios más altos reducen la probabilidad de ventas altas.

Normalización: Usar X_train_norm y X_test_norm (normalizados con MinMaxScaler) asegura que las variables contribuyan equitativamente al modelo.
-->
# Prueba de hipótesis


**Supuestos**

[ regresión logistica](https://julius.ai/articles/decoding-the-core-assumptions-of-logistic-regression)


# Modelos y parámetros
    
Detalle los modelos utilizados, incluyendo la selección de parámetros específicos e hiperparámetros. 
Si ha realizado ajustes de hiperparámetros, proporcione una descripción general de su estrategia de búsqueda 
y la justificación de los valores seleccionados.
    


Detalle del modelo utilizado: Regresión logística 

Descripción:

Predice la probabilidad de que una observación pertenezca a la clase (0,1) usando la función sigmoide:


$P(t) =\frac{1}{1 + e^{-(w^T X + b)}}$

Entrenado mediante descenso de gradiente para minimizar la función de costo de entropía cruzada binaria (log-loss):

$J(w,b)=-\frac{1}{m}\sum_{i=1}^m[y_ilog⁡(y_i)+(1−yi)log⁡(1−y_i)]$






Datos estandarizados (StandardScaler) para asegurar que todas las características contribuyan equitativamente.

Selección de parámetros e hiperparámetros:
Parámetros del modelo:
w: Vector de pesos inicializado en ceros (np.zeros(n)), donde n es el número de características (21 numericas en este caso).

b: Sesgo inicializado en 0.

Hiperparámetros:
learning_rate=0.01: Tasa de aprendizaje para el descenso de gradiente.

epochs=1000: Número de iteraciones para entrenar el modelo.

Umbral de clasificación: 0.5 (en y_pred_class = (y_pred >= 0.5).astype(int)).


#  Código y flujo de trabajo computacional

**Especificaciones del entorno**

  
    
    requirements.txt 

<!--
#  Resultados e interpretaciones

Es crucial no solo presentar los resultados, sino también interpretarlos en el contexto del problema original. Esto implica tener:

- **Resumen de los hallazgos**
    
    : un resumen conciso de las ideas clave derivadas del análisis, idealmente presentadas mediante gráficos y tablas.
    
- **Significancia estadística**[la prueba de hipótesis](https://www.statology.org/hypothesis-testing/)
    
    : Para
    
    , indique claramente los valores p y explique sus implicaciones. Además, proporcione intervalos de confianza para los parámetros estimados, cuando corresponda.
    
- **Limitaciones del análisis**
    
    : Mencione cualquier limitación inherente a sus modelos, por ejemplo, sobreajuste, falta de generalización o variables omitidas.
    
- **Visualización**[y estén claramente etiquetados para indicar lo que se muestra](https://www.statology.org/5-data-visualization-techniques-to-make-your-findings-stand-out/)
    
    : Las representaciones visuales suelen ser más intuitivas. Asegúrese de que todos los gráficos y diagramas incluyan descripciones

-->

-------------------------------------------------------------

