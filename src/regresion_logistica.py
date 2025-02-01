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

    data = data[data['symbol'] == symbol] # Reemplaza 'btc/usd' con el símbolo deseado

    # Crear una nueva columna para el precio del día siguiente
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
    X = selector.fit_transform(X, y)
    selected_features = np.array(features)[selector.get_support()]

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    if len(selected_features) < len(features):
        print("\nCoeficientes de las características seleccionadas:")
        coefficients = best_model.coef_[0]
        for feature, coefficient in zip(selected_features, coefficients):
            print(f"{feature}: {coefficient:.4f}")
    else:
        print("\nCoeficientes del Modelo:")
        coefficients = best_model.coef_[0]
        for feature, coefficient in zip(features, coefficients):
            print(f"{feature}: {coefficient:.4f}")