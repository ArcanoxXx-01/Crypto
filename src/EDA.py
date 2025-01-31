import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from read_data import read_data
import numpy as np

def analice(var: str, days):
    # Gráfico de variable a lo largo del tiempo
    data= read_data(days, [var])
    plt.figure(figsize=(12, 6))

    plt.plot(data[var], label=var, color='blue')
    plt.title(f'{var} de la Criptomoneda a lo Largo del Tiempo')
    plt.xlabel('Fecha')
    plt.ylabel('Precio (USD)')
    plt.legend()
    plt.show()

def histograma(var: str, days):
    data= read_data(days, [var])
    # Histograma de una variable
    plt.figure(figsize=(12, 6))
    sns.histplot(data[var], bins=50, kde=True)
    plt.title(f'Distribución de {var}')
    plt.xlabel('Precio (USD)')
    plt.ylabel('Frecuencia')
    plt.show()

    # Boxplot de precios de cierre
    plt.figure(figsize=(12, 6))
    sns.boxplot(y=data[var])
    plt.title(f'Boxplot de {var}')
    plt.ylabel('Precio (USD)')
    plt.show()

def correlation(days):
    # Mapa de calor para ver la correlación entre variables
    data= read_data(days, ['open', 'high', 'low', 'close', 'volume usd', 'volume cripto', 'volatility', 'average'])
    correlation_matrix = data[['open', 'high', 'low', 'close', 'volume usd', 'volume cripto', 'volatility', 'average']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Mapa de Calor de Correlaciones')
    plt.show()

def attipic_values(days, var1 , var2):
    # Gráfico de dispersión para detectar valores atípicos en var1 y var2
    data= read_data(days, [var1, var2])
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=data, x=var1, y=var2)
    plt.title(f'{var1} vs {var2}')
    plt.xlabel(f'{var1}')
    plt.ylabel(f'{var2}')
    plt.xscale('log')  # Escala logarítmica para mejor visualización
    plt.yscale('log')
    plt.show()    


def describe(days, symbol):
    # Cargar los datos
    data = read_data(days, ['open', 'close', 'high', 'low', 'volatility', 'volume usd', 'volume cripto', 'average', 'symbol'])
    
    # Verificar si la columna 'symbol' está presente
    if 'symbol' not in data.columns:
        print("Error: La columna 'symbol' no está en los datos cargados.")
        return

    # Filtrar por símbolo de criptomoneda
    data_filtered = data[data['symbol'] == symbol]  # Cambia aquí el filtrado

    # Verificar si hay datos disponibles para el símbolo
    if data_filtered.empty:
        print(f"No se encontraron datos para la criptomoneda: {symbol}")
        return

    # Cálculo de estadísticas descriptivas
    mean_close = data_filtered['close'].mean()
    median_close = data_filtered['close'].median()
    mode_close = data_filtered['close'].mode().values[0]

    mean_high = data_filtered['high'].mean()
    median_high = data_filtered['high'].median()
    mode_high = data_filtered['high'].mode().values[0]

    mean_low = data_filtered['low'].mean()
    median_low = data_filtered['low'].median()
    mode_low = data_filtered['low'].mode().values[0]

    mean_volume_usd = data_filtered['volume usd'].mean()
    median_volume_usd = data_filtered['volume usd'].median()
    mode_volume_usd = data_filtered['volume usd'].mode().values[0]

    mean_volume_crypto = data_filtered['volume cripto'].mean()
    median_volume_crypto = data_filtered['volume cripto'].median()
    mode_volume_crypto = data_filtered['volume cripto'].mode().values[0]

    # Imprimir resultados
    print(f"Estadísticas de Precio de Cierre para {symbol}:")
    print(f"Media: {mean_close}, Mediana: {median_close}, Moda: {mode_close}\n")

    print(f"Estadísticas de Precio Máximo para {symbol}:")
    print(f"Media: {mean_high}, Mediana: {median_high}, Moda: {mode_high}\n")

    print(f"Estadísticas de Precio Mínimo para {symbol}:")
    print(f"Media: {mean_low}, Mediana: {median_low}, Moda: {mode_low}\n")

    print(f"Estadísticas de Volumen USD para {symbol}:")
    print(f"Media: {mean_volume_usd}, Mediana: {median_volume_usd}, Moda: {mode_volume_usd}\n")

    print(f"Estadísticas de Volumen Cripto para {symbol}:")
    print(f"Media: {mean_volume_crypto}, Mediana: {median_volume_crypto}, Moda: {mode_volume_crypto}\n")   

    # Gráficos de barras para media, mediana y moda
    labels = ['Cierre', 'Máximo', 'Mínimo', 'Vol. Cripto']
    mean_values = [mean_close, mean_high, mean_low, mean_volume_crypto]
    median_values = [median_close, median_high, median_low, median_volume_crypto]
    mode_values = [mode_close, mode_high, mode_low, mode_volume_crypto]
    # Crear el gráfico
    plt.figure(figsize=(14, 6))

    x = np.arange(len(labels))
    # Gráfico de media
    plt.bar(x, mean_values, width=0.2, label='Media', color='lightblue', align='center')
    # Gráfico de mediana
    plt.bar([p + 0.2 for p in x], median_values, width=0.2, label='Mediana', color='lightgreen', align='center')
    # Gráfico de moda
    plt.bar([p + 0.4 for p in x], mode_values, width=0.2, label='Moda', color='salmon', align='center')

    plt.xlabel('Variables')
    plt.ylabel('Valores')
    plt.title(f'Comparación de Media, Mediana y Moda de Variables de Criptomonedas para {symbol}')
    plt.xticks([p + 0.2 for p in x], labels)
    plt.legend()
    plt.grid(axis='y')  # Añadir una cuadrícula en el eje y para mejor visualización
    plt.show()
