from sklearn.linear_model import LinearRegression
import numpy as np
from read_data import read_data
import matplotlib.pyplot as plt
import statsmodels.api as sm

def simple_regresion(days: int, var1, var2):
    # Cargar el CSV
    data = read_data(days, [var2, var1])

    # Asegúrate de eliminar valores nulos
    data = data.dropna(subset=[var2, var1])

    # Definir variable independiente y dependiente
    X = data[var1]  # variable independiente
    Y = data[var2]  # variable dependiente

    # Agregar constante a la variable independiente
    X = sm.add_constant(X)

    # Ajustar el modelo de regresión
    model = sm.OLS(Y, X).fit()

    # Ver el resumen del modelo
    print(model.summary())

    # Gráfico de dispersión y línea de regresión
    plt.figure(figsize=(12, 6))
    plt.scatter(data[var1], data[var2], color='blue', alpha=0.6, label='Datos Observados')
    plt.plot(data[var1], model.predict(X), color='red', label='Línea de Regresión', linewidth=2)
    plt.title(f'Regresión Simple: {var1} vs {var2}')
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.legend()
    plt.grid()
    plt.show()
 

