import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

"""### **CHAT BOT RECOMENDACION MEDICAMENTO**

El programa funciona a partir de una recolección de datos de un estudio de un grupo de pacientes, que padecen la misma enfermedad. Durante el tratamiento, cada paciente respondio a una de 5 medicaciones: Droga A, Droga B, Droga C, Droga X y Droga Y.

El modelo que realizamos busca encontrar el medicamento apropiado para un próximo paciente con la misma enfermedad. El conjunto de características de cada paciente incluye: Edad, Sexo, Presión Sanguínea, Colesterol, Potasio, Sodio, y Na_to_K que se obtiene al dividir el valor de sodio
por el valor de potasio
"""

#leer DataSet
df = pd.read_csv('drug200.csv', sep=';')

df.head(15)
#cantidad de registros = 200
#caracteristicas = 6
#variables independientes = Age, Sex, BP, Cholesterol, Na_to_K
#variable de respuesta = Drug

#se crea un tensor X para todas las varibales independientes con las que se va a realizar la predicción
#se crea un tensor y para guardar la variable dependiente de salida predictora
x_tensor = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']
y_tensor = ['Drug']
#llevar los datos del DataFrame a los tensores creados
X = df[x_tensor].values
y = df[y_tensor].values

"""Directamente sobre el DF

<!--

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
y = my_data['Drug'].values

-->
"""

# Conversión del atributo sexo en una variable numérica codificada.
#le_sex.fit define las variables categoricas que se van a transformar y las guarda en le_sex
#le_sex.transform transforma esas variables categoricas en valores numéricos
#le_sex.transform(X[:,1]) indica que tome todas las filas del arreglo bidimensional con este argumento ":" y la columna "1" que es el numero de columna de la variable "sexo"
#X[:,1] = esa transformación que de acuerdo a lo anterior, ahora son numeros los reemplaza en todas las filas el vector X bidimensional de la columna 1 que es el sexo
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])

#conversión del atributo BP a una variable numéricada codificada
le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

#conversión del atributo Cholesterol a una variable numérica codificada
le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

#división del conjuto de datos en un 30% para pruebas y 70% para entrenamiento
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    random_state = 3)

#verificar forma de los tensores creados
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

#crear el arbol de decisión criterio de impureza "entropy"
#tamaño del arbol 4 niveles de acuerdo al árbol de decisión del dataset
from sklearn.tree import DecisionTreeClassifier
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

#entrenamiento del modelo con los datos de entrenamiento almacenados en los vectores X_train, y_train
drugTree.fit(X_train, y_train)

#realizar predicción con el método predict de la clase DecisionTreeClasifier
predTree = drugTree.predict(X_test)

#evaluar el modelo con la métrica de

def precision():
 
    imp =print("Precisión del modelo con los datos de pruebas y_test", metrics.accuracy_score(y_test, predTree),"%")
    return imp
    ##predTree

"""**VISUALIZACION DEL ARBOL**

"""

# Commented out IPython magic to ensure Python compatibility.
# Importar la clase que permite generar la gráfica
from sklearn import tree

# Importar librería para gráficos
import matplotlib.pyplot as plt
# %matplotlib inline

# Obtener como una lista, el nombre de las columnas del dataset
featureNames = df.columns[0:5].tolist()

# Obtener como una lista, los posibles valores que toma
# la variable de respuesta
classNames = targetNames = df["Drug"].unique().tolist()

# Crear los parámetros de visualización de la imagen
fig = plt.subplots(figsize=(10, 10))

# Generar y mostrar la gráfica del árbol
tree.plot_tree(drugTree, feature_names=featureNames, class_names=classNames,
               filled=True)
#plt.show()


# Importar la clase para generar el árbol
from sklearn.tree import export_text
t = export_text(drugTree, feature_names=featureNames)
#print(t)

"""**PREDICCION**"""

#X_test[:20], y_test[:20]
#df.head()

#datos leidos por pantalla
import numpy as np
#crear un arreglo con los datos de un paciente
datos = np.array([[67,'M','NORMAL','NORMAL',10.898]])
datos

#conversión de las variable "sexo" categórica en numérica
datos[0,1] = 0 if datos[0,1] == 'F' else 1
#conversión de las variable "BP" categórica en numérica
datos[0,3] = 0 if datos[0,3] == 'HIGH' else 1
#conversión de las variable "Cholesterol" categórica en numérica
if datos[0,2] == 'HIGH':
  datos[0,2] = 0
elif datos[0,2] == 'LOW':
    datos[0,2] = 1
else:
    datos[0,2] = 2
datos

prediccion = drugTree.predict(datos)
#prediccion

def predecir_medicamento(datos_paciente):
    # Aquí va tu código para predecir el medicamento recomendado
    # Debes procesar los datos del paciente según como lo haces en tu código actual
    # Y luego usar el modelo `drugTree` para predecir el medicamento
    
    # Por ejemplo:
    # datos_paciente es un array de numpy con las características del paciente
    prediccion = drugTree.predict(datos_paciente)
    return prediccion

# Tu código actual para cargar datos, entrenar el modelo, etc.