import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from read_data import read_data

def mediacion_simple(days):
    # Cargar el CSV
    data = read_data(days, ['average', 'volume usd', 'volatility'])

    # Asegúrate de eliminar valores nulos
    data = data.dropna(subset=['average', 'volume usd', 'volatility'])

    # Definición de las variables
    Y = data['average']  # 'average' como variable dependiente (precio promedio)
    X = data['volume usd']  # 'volume usd' como variable independiente
    M = data['volatility']  # 'volatility' como variable mediadora

    # Análisis de mediación
    print("Analizando cómo el volumen de trading (variable independiente) influye en el precio promedio (dependiente) a través de la volatilidad (mediadora).")

    # 1. Regresión de la variable mediadora (volatilidad) en función de la variable independiente (volumen)
    X_M = sm.add_constant(X)
    model_M = sm.OLS(M, X_M).fit()

    # 2. Regresión de la variable dependiente (precio promedio) en función de la variable mediadora y la independiente
    X_Y = sm.add_constant(data[['volume usd', 'volatility']])
    model_Y = sm.OLS(Y, X_Y).fit()

    # Resultados de ambos modelos
    print("\nResultados del modelo de mediación para la variable mediadora (volatilidad):")
    print(model_M.summary())

    print("\nResultados del modelo de mediación para la variable dependiente (precio promedio):")
    print(model_Y.summary())

    # Gráfico
    plt.figure(figsize=(12, 6))

    # Gráfico de dispersión de la variable independiente contra la variable mediadora
    sns.scatterplot(x='volume usd', y='volatility', data=data, alpha=0.6)
    plt.title('Volumen de Trading vs Volatilidad')
    plt.xlabel('Volumen de Trading (USD)')
    plt.ylabel('Volatilidad')
    plt.grid()

    # Agregar línea de regresión para mediadora
    sns.regplot(x='volume usd', y='volatility', data=data, scatter=False, color='red')
    plt.show()

    # Gráfico de dispersión de la variable mediadora contra la variable dependiente
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='volatility', y='average', data=data, alpha=0.6)
    plt.title('Volatilidad vs Precio Promedio')
    plt.xlabel('Volatilidad')
    plt.ylabel('Precio Promedio')
    plt.grid()

    # Agregar línea de regresión para variable dependiente
    sns.regplot(x='volatility', y='average', data=data, scatter=False, color='red')
    plt.show()
