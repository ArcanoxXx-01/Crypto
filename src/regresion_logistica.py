import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from read_data import read_data


def logistic_regresion_cripto(days, symbol):
    """
    Realiza una regresión logística para predecir el movimiento del precio de una criptomoneda específica.

    Args:
        days (int): Número de días de datos a utilizar.
        symbol (str): Símbolo de la criptomoneda (ej. 'BTC/USD').
    """
    features = ['open', 'high', 'low', 'volatility', 'average', 'close', 'volume usd', 'volume cripto']

    data = read_data(days, features + ['symbol'])
    data = data[data['symbol'] == symbol]  # Reemplaza con el símbolo deseado

    # Crear la columna para el precio del día siguiente
    data['next_day_close'] = data['close'].shift(-1)

    # Crear la variable objetivo binaria
    data['target'] = np.where(data['next_day_close'] > data['close'], 1, 0)

    # Eliminar filas con NaN
    data.dropna(inplace=True)

    # Estandarizar las características
    scaler = StandardScaler()
    X = scaler.fit_transform(data[features])
    y = data['target']

    # Selección de características (opcional)
    selector = SelectKBest(f_classif, k=5)
    X_selected = selector.fit_transform(X, y)
    selected_features = np.array(features)[selector.get_support()]

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Ajuste de hiperparámetros
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
    grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Predicciones
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # Evaluación
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Resultados
    print("Mejores hiperparámetros:", grid_search.best_params_)
    print("Accuracy:", accuracy)
    print("AUC-ROC:", roc_auc)
    print("Matriz de Confusión:\n", conf_matrix)
    print("\nReporte de Clasificación:\n", class_report)

    # Coeficientes
    print("\nCoeficientes del Modelo:")
    coefficients = best_model.coef_[0]
    for feature, coefficient in zip(selected_features, coefficients):
        print(f"{feature}: {coefficient:.4f}")

    # Predicción para el siguiente día
    # Obtener los datos más recientes para la predicción
    last_data = data[features].iloc[-1].values.reshape(1, -1)  # Últimos valores de las características

    # Estandarizar los datos de entrada para la predicción usando el mismo scaler
    last_data_scaled = scaler.transform(last_data)

    # Utilizar el selector para transformar la entrada de acuerdo a las características seleccionadas
    last_data_selected = selector.transform(last_data_scaled)

    # Realizar la predicción
    prediction = best_model.predict(last_data_selected)
    prediction_prob = best_model.predict_proba(last_data_selected)[:, 1]

    # Resultado de la predicción
    print(f"\nPredicción para mañana: {'Aumento' if prediction[0] == 1 else 'No Aumento'}")
    print(f"Probabilidad de aumento: {prediction_prob[0]:.2f}")