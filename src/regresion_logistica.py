from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from read_data import read_data
import numpy as np


def logistic_regresion():
    # Seleccionar los datos de relevancia
    features = ['open', 'high', 'low', 'volatility', 'average', 'close', 'volume usd', 'volume cripto']

    # Leer datos
    data= read_data(100, features)

    # Crear una nueva columna para el precio del día siguiente
    data['next_day_close'] = data['close'].shift(-1)

    # Crear la variable de respuesta binaria: 1 si el precio de cierre del día siguiente es mayor, 0 en caso contrario
    data['target'] = np.where(data['next_day_close'] > data['close'], 1, 0)

    # Eliminar filas con valores NaN resultantes de la operación shift
    data.dropna(inplace=True)

    # Seleccionar características para el modelo
    X = data[features]
    y = data['target']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo de regresión logística
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Hacer predicciones
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Imprimir los resultados
    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    # Examinar los coeficientes del modelo para entender la importancia de cada característica
    coefficients = model.coef_[0]
    features_coefficients = zip(features, coefficients)

    print("\nFeature Coefficients:")
    for feature, coefficient in features_coefficients:
        print(f"{feature}: {coefficient:.4f}")