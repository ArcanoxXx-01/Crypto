import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from read_data import read_data

def ANOVA(days):
    # Cargar el CSV
    data = read_data(days, ['open', 'close', 'symbol'])

    # Asegúrate de eliminar valores nulos en columnas 'open' y 'close'
    data = data.dropna(subset=['open', 'close', 'symbol'])

    # Calcular la variable de retornos
    data['returns'] = (data['close'] - data['open']) / data['open']

    # Dividir los datos por grupos basados en 'symbol'
    grouped_returns = [group['returns'].values for name, group in data.groupby('symbol')]

    # Hipótesis
    print("Hipótesis nula (H0): La media de los retornos es igual para todas las criptomonedas.")
    print("Hipótesis alternativa (H1): Al menos una de las medias de los retornos es diferente.")

    # Realizar ANOVA de un solo factor
    anova_result = stats.f_oneway(*grouped_returns)

    print(f"Estadístico F: {anova_result.statistic}, Valor p: {anova_result.pvalue}")

    # Interpretación de resultados
    if anova_result.pvalue < 0.05:
        print("Se rechaza la hipótesis nula: hay diferencias significativas entre las medias de los retornos de las criptomonedas.")
    else:
        print("No hay suficiente evidencia para rechazar la hipótesis nula.")

    # Gráfico de caja (boxplot) para visualizar los retornos por criptomoneda
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='symbol', y='returns', data=data)
    plt.title('Comparación de los Retornos por Criptomoneda')
    plt.xlabel('Criptomoneda')
    plt.ylabel('Retornos')
    plt.grid()
    plt.show()