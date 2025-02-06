import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from read_data import read_data

def ANOVA(days, var):
    # Cargar el CSV
    data = read_data(days, [var, 'symbol'])

    data = data.dropna(subset=[var, 'symbol'])

    # Dividir los datos por grupos basados en 'symbol'
    grouped_returns = [group[var].values for name, group in data.groupby('symbol')]

    # Hipótesis
    print(f"Hipótesis nula (H0): La media de la variable {var} es igual para todas las criptomonedas.")
    print(f"Hipótesis alternativa (H1): Al menos una de las medias de la variable {var} es diferente.")

    # Realizar ANOVA de un solo factor
    anova_result = stats.f_oneway(*grouped_returns)

    print(f"Estadístico F: {anova_result.statistic}, Valor p: {anova_result.pvalue}")

    # Interpretación de resultados
    if anova_result.pvalue < 0.05:
        print(f"Se rechaza la hipótesis nula: hay diferencias significativas entre las medias de la variable {var} de las criptomonedas.")
    else:
        print("No hay suficiente evidencia para rechazar la hipótesis nula.")

    # Gráfico de caja (boxplot) para visualizar los retornos por criptomoneda
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='symbol', y=var, data=data)
    plt.title(f'Comparación de {var} por Criptomoneda')
    plt.xlabel('Criptomoneda')
    plt.ylabel(f'{var}')
    plt.grid()
    plt.show()