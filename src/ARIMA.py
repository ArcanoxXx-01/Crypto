import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from read_data import read_data

def arima(var, days, symbol=None):
    # Cargar los datos
    data = read_data(days, [var, 'symbol', 'date'])
    
    # Filtrar los datos según el símbolo
    if symbol:
        data = data[data['symbol'] == symbol]

    data['date'] = pd.to_datetime(data['date'])
    
    # Establecer la columna 'date' como índice
    data.set_index('date', inplace=True)

    # Verificar que haya datos suficientes
    if len(data) < 2:
        print("No hay suficientes datos para realizar el análisis.")
        return
    
    # Visualización de la serie temporal
    plt.figure(figsize=(10, 6))
    plt.plot(data[var], label='Datos Históricos')
    plt.title(f'Serie Temporal de {var} para {symbol}')
    plt.xlabel('Fecha')
    plt.ylabel(var)
    plt.legend()
    plt.show()

    # Ajuste del modelo ARIMA
    p = 2  # Orden del modelo AR (ajustar según análisis ACF/PACF)
    d = 3  # Diferenciación
    q = 4  # Orden del modelo MA (ajustar según análisis ACF/PACF)

    model = ARIMA(data[var], order=(p, d, q))
    model_fit = model.fit()

    # Resumen del modelo
    print(model_fit.summary())
    
    # Pronóstico
    forecast_steps = 10  # Cambiar según necesidades
    forecast = model_fit.forecast(steps=forecast_steps)
    print(f"Pronóstico para los próximos {forecast_steps} períodos:")
    print(forecast)

    # Visualización del pronóstico
    plt.figure(figsize=(10, 6))
    plt.plot(data[var], label='Datos Históricos')
    plt.plot(np.arange(len(data), len(data) + forecast_steps), forecast, label='Pronóstico', color='red')
    plt.title(f'Pronóstico ARIMA de {var} para {symbol}')
    plt.xlabel('Periodo')
    plt.ylabel(var)
    plt.legend()
    plt.show()

# Ejemplo de uso:
# arima('volatility',30, 'BTC/USD' )
