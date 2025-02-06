import pandas as pd
from scipy import stats
from read_data import read_data
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta, datetime

def test_wilcoxon(days, var, date: str, type= None):
    # Cargar el CSV
    data= read_data(days ,['date', 'symbol']+[var])

    if type:
        data= data[data['symbol']==type]

    DATE= datetime.strptime(date,'%Y-%m-%d').date()

    date_before= (DATE- timedelta(days=25)).strftime("%Y-%m-%d")
    date_after= (DATE+ timedelta(days=25)).strftime("%Y-%m-%d")    
    # Comparar variable return antes y después de un evento (suponiendo que tienes dos subconjuntos)
    before_event = data[var][(data['date'] < date) & (data['date']>= date_before)]
    after_event = data[var][(data['date'] >= date) & (data['date']<= date_after)]

    # Asegúrate de que ambos conjuntos tengan la misma longitud
    min_length = min(len(before_event), len(after_event))

    # Truncar ambos conjuntos al mínimo común
    before_event = before_event.iloc[:min_length]
    after_event = after_event.iloc[:min_length]

    # Hipótesis
    print(f"Hipótesis nula (H0): la mediana de la variable {var} antes del evento es igual a la mediana de la variable {var} después del evento.")
    print(f"Hipótesis alternativa (H1): la mediana de la variable {var} antes del evento no es igual a la mediana de la variable {var} después del evento.")

    wilcoxon_statistic, wilcoxon_p_value = stats.wilcoxon(before_event, after_event)
    print(f"Prueba de Wilcoxon: estadístico = {wilcoxon_statistic}, valor p = {wilcoxon_p_value}")

    # Interpretación
    if wilcoxon_p_value < 0.05:
        print(f"Se rechaza la hipótesis nula: la variable {var} antes y después del evento es significativamente diferente.")
    else:
        print("No hay suficiente evidencia para rechazar la hipótesis nula.")

    # Gráfica de boxplot para visualizar "antes" y "después"
    plt.figure(figsize=(10, 6))
    plt.boxplot([before_event, after_event], labels=['Antes del Evento', 'Después del Evento'])
    plt.title(f'Comparación de {var} Antes y Después del Evento')
    plt.ylabel(var)
    plt.grid()
    plt.show()    


def test_kruskal_wallis(days, cripto1, cripto2, cripto3):
    # Supongamos que estás comparando la volatilidad de diferentes criptomonedas
    data= read_data(days , ['volatility', 'symbol'])
    group1 = data[data['symbol'] == cripto1]['volatility']
    group2 = data[data['symbol'] == cripto2]['volatility']
    group3 = data[data['symbol'] == cripto3]['volatility']
    
    # Hipótesis
    print("Prueba de Kruskal-Wallis")
    print("Hipótesis nula (H0): las medianas de la volatilidad de diferentes grupos son iguales.")
    print("Hipótesis alternativa (H1): al menos una de las medianas de la volatilidad de diferentes grupos es diferente.")

    # Supongamos que estás comparando la volatilidad de diferentes criptomonedas
    kruskal_statistic, kruskal_p_value = stats.kruskal(group1, group2, group3)
    print(f"Prueba de Kruskal-Wallis: estadístico = {kruskal_statistic}, valor p = {kruskal_p_value}")

    # Interpretación
    if kruskal_p_value < 0.05:
        print("Se rechaza la hipótesis nula: hay diferencias significativas en la volatilidad entre las criptomonedas.")
    else:
        print("No hay suficiente evidencia para rechazar la hipótesis nula.")
    # Gráfica de boxplot para visualizar la volatilidad de las criptomonedas
    plt.figure(figsize=(10, 6))
    plt.boxplot([group1, group2, group3], labels=[cripto1, cripto2, cripto3])
    plt.title('Volatilidad de Criptomonedas')
    plt.ylabel('Volatilidad')
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()


def test_spearman( days, var1, var2):

    data= read_data(days , [var1, var2])

    # Hipótesis
    print("Correlación de Spearman")
    print(f"Hipótesis nula (H0): no hay correlación entre la variable {var1} y {var2}.")
    print(f"Hipótesis alternativa (H1): hay una correlación entre la variable {var1} y {var2}.")

    # Evaluar la correlación entre el la variable returns y la volatilidad
    spearman_corr, spearman_p_value = stats.spearmanr(data[var1], data[var2])
    print(f"Correlación de Spearman: coeficiente = {spearman_corr}, valor p = {spearman_p_value}")

    # Interpretación
    if spearman_p_value < 0.05:
        print(f"Hay una correlación significativa entre el la variable {var1} y la {var2}.")
    else:
        print("No hay suficiente evidencia para concluir que existe una correlación.")      


def test_mannwhitneyu(days, var, type1, type2):

    data= read_data(days, [var, 'symbol'])
    # Dividir los datos en dos grupos, por ejemplo, comparar la volatilidad de dos criptomonedas
    group1 = data[data['symbol'] == type1][var]
    group2 = data[data['symbol'] == type2][var]

    # Hipótesis
    print("Prueba de Mann-Whitney U")
    print(f"Hipótesis nula (H0): las distribuciones de la Variable {var} de {type1} y {type2} son iguales.")
    print(f"Hipótesis alternativa (H1): las distribuciones de la Variable {var} de {type1} y {type2} son diferentes.")

    # Realizar la prueba de Mann-Whitney U
    mannwhitney_statistic, mannwhitney_p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    print(f"Prueba de Mann-Whitney U: estadístico = {mannwhitney_statistic}, valor p = {mannwhitney_p_value}")

    # Interpretación
    if mannwhitney_p_value < 0.05:
        print(f"Se rechaza la hipótesis nula: hay una diferencia significativa en la Variable {var} entre {type1} y {type2}.")
    else:
        print("No hay suficiente evidencia para rechazar la hipótesis nula.")

    # Gráfica de las distribuciones
    plt.figure(figsize=(12, 6))

    # Histograma de las distribuciones
    sns.histplot(group1, bins=30, color='blue', label=type1, stat='density', kde=True, alpha=0.6)
    sns.histplot(group2, bins=30, color='orange', label=type2, stat='density', kde=True, alpha=0.6)

    plt.title(f'Distribuciones de Variable {var} de {type1} y {type2}')
    plt.xlabel(f'Variable {var}')
    plt.ylabel('Densidad')
    plt.legend()
    plt.grid()
    plt.show()    
   