
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from read_data import read_data


# 1. Cargar los datos
# df = pd.read_csv('ruta_a_tus_datos.csv')
# df['fecha'] = pd.to_datetime(df['fecha'])
# df.set_index('fecha', inplace=True)

def SARIMA (var , days):
    df = read_data(days, [var,'date'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # 2. Visualización de la serie temporal (similar al ejemplo anterior)
    plt.figure(figsize=(10, 6))
    plt.plot(df['valor'])
    plt.title('Serie Temporal')
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.show()

    # 3. Ajuste del modelo SARIMA
    p = 1    # Orden del modelo AR
    d = 1    # Diferenciación
    q = 1    # Orden del modelo MA
    P = 1    # Orden estacional AR
    D = 1    # Estacional diferenciación
    Q = 1    # Orden estacional MA
    s = 12   # Periodo estacional (por ejemplo, 12 para datos mensuales con estacionalidad anual)

    model = SARIMAX(df['valor'], order=(p, d, q), seasonal_order=(P, D, Q, s))
    model_fit = model.fit()

    # 4. Resumen del modelo
    print(model_fit.summary())

    # 5. Pronóstico
    forecast = model_fit.forecast(steps=10)  # Pronosticar los próximos 10 periodos
    print(forecast)

    # 6. Visualización del pronóstico
    plt.figure(figsize=(10, 6))
    plt.plot(df['valor'], label='Datos Históricos')
    plt.plot(np.arange(len(df), len(df) + 10), forecast, label='Pronóstico', color='red')
    plt.title('Pronóstico SARIMA')
    plt.xlabel('Periodo')
    plt.ylabel('Valor')
    plt.legend()
    plt.show()