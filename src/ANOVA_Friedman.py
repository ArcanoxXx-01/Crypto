from read_data import read_data
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def ANOVA_Friedman(days):
    # Cargar los datos
    data = read_data(days, ['date', 'symbol', 'open', 'close'])

    # Asegúrate de que 'date' sea del tipo datetime
    data['date'] = pd.to_datetime(data['date'])

    # Calcular la variable de retornos
    data['returns'] = (data['close'] - data['open']) / data['open']

    # Crear una nueva columna para agrupar por períodos de 10 días
    data['period'] = data['date'].dt.to_period('10D')

    # Agrupar los datos por períodos y símbolo y calcular la mediana de los retornos
    grouped_returns = data.groupby(['period', 'symbol'])['returns'].median().reset_index()

    # Pivotar el DataFrame para que cada símbolo tenga su propia columna
    df_friedman = grouped_returns.pivot(index='period', columns='symbol', values='returns')

    # Asegúrate de que hay suficientes datos
    if df_friedman.shape[1] < 2:
        print("Necesitas al menos 2 grupos (criptomonedas) para realizar ANOVA de Friedman.")
        return

    # Imprimir el DataFrame reformateado
    print(df_friedman)

    # Ejecutar el ANOVA de Friedman
    stat, p = stats.friedmanchisquare(*[df_friedman[col].dropna() for col in df_friedman.columns])

    print(f'Estadístico de Friedman: {stat}, Valor p: {p}')

    # Interpretación de resultados
    if p < 0.05:
        print('Se rechaza la hipótesis nula: hay diferencias significativas entre las medias de los retornos de las criptomonedas.')
    else:
        print('No hay suficiente evidencia para rechazar la hipótesis nula.')

    # Gráfico de caja (boxplot) para visualizar los retornos por criptomoneda
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_friedman)
    plt.title('Comparación de los Retornos por Criptomoneda en Períodos de 10 Días')
    plt.xlabel('Criptomoneda')
    plt.ylabel('Retornos')
    plt.grid()
    plt.show()
