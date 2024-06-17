# Pandas para manejar datos.
# NumPy para operaciones numéricas. 
# Scikit-learn para el análisis de texto y el entrenamiento del modelo.
# NLTK para el procesamiento de lenguaje natural.
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import joblib


# Carga de datos de entrenamiento
print('Iniciando la aplicación...\n')
train_data = pd.read_csv('fitData.csv')

# Preprocesamiento de texto
# Este bloque descarga los paquetes de tokenización y stopwords de NLTK, y define una función preprocess_text que:
# Tokeniza el texto en palabras individuales
# Convierte el texto a minúsculas
# Elimina las stopwords (palabras comunes como "de", "la", etc.)
# Devuelve el texto preprocesado
# Luego, aplica esta función a cada texto en la columna text de los datos de entrenamiento.
print('Descargando paquetes...\n')
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('spanish'))
print('Eliminando palabras comunes...\n')
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)
print('Aplicando funciones...\n')
train_data['text'] = train_data['text'].apply(preprocess_text)

# Este bloque crea un objeto TfidfVectorizer para vectorizar el texto, y lo utiliza para transformar los textos en vectores numéricos. 
# El parámetro max_features se establece en 10000 para limitar el número de características.
print('Transformando texto en vectores numéricos y limitando características...\n')
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(train_data['text'])
y = train_data['label']

#Este bloque divide los datos en conjuntos de entrenamiento y prueba, con una proporción de 90% para entrenamiento y 10% para prueba.
print('Dividiendo datos de entrenamiento - prueba (90% - 10%)...\n')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

try:
    # Cargar el modelo entrenado desde el archivo
    model = joblib.load('textClass/data/data.joblib')
except FileNotFoundError:
    # Si no existe el archivo, entrenar la red neuronal
    print('Entrenando la red neuronal...')
    model = LogisticRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, 'data/data.joblib')
    print('Red neuronal entrenada con éxito.')

# Este bloque: Pide al usuario que ingrese un texto para predecir 
# Predice la etiqueta para el texto de prueba utilizando el modelo entrenado 
# Imprime la precisión y el informe de clasificación para el conjunto de prueba 
# Preprocesa el texto ingresado por el usuario
# Vectoriza el texto preprocesado
# Predice la etiqueta para el texto ingresado utilizando el modelo entrenado
# Imprime la predicción

nT=input("Ingrese el texto que desea predecir: ")

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report: \n")
print('La siguiente advertencia se debe a que la precisión se calcula como el número de verdaderos positivos dividido por el número de predicciones positivas.')
print('Si no hay predicciones positivas para una etiqueta, la precisión se vuelve indefinida (0/0), lo que lleva a la advertencia.\n')
print(classification_report(y_test, y_pred))

new_text = nT
print(new_text)
new_text = preprocess_text(new_text)
new_text_vector = vectorizer.transform([new_text])
prediction = model.predict(new_text_vector)
print("Predicción:", prediction[0])

#precision: La precisión es la proporción de verdaderos positivos (TP) entre todos los positivos predichos (TP + FP).
# # En otras palabras, es la probabilidad de que un ejemplo sea verdaderamente positivo cuando el modelo lo predice como positivo.

# recall: La recall es la proporción de verdaderos positivos (TP) entre todos los positivos reales (TP + FN). En otras palabras, 
# # es la proborción de ejemplos positivos reales que el modelo detecta correctamente.

# f1-score: El f1-score es la media armónica de la precisión y la recall. Es una métrica que combina la
# # precisión y la recall para proporcionar una visión general del desempeño del modelo.

# support: El soporte es el número de ejemplos en la clase correspondiente.

#Filas adicionales:

# accuracy: La precisión global del modelo, que es la proporción de ejemplos predichos correctamente entre todos los ejemplos.

# macro avg: La media de las métricas (precisión, recall y f1-score) para todas las clases.

# weighted avg: La media ponderada de las métricas (precisión, recall y f1-score) para todas las clases, donde el peso es el 
# # soporte de cada clase.