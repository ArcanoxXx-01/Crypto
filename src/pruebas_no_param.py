import pandas as pd
from scipy import stats
from read_data import read_data
import matplotlib.pyplot as plt
import seaborn as sns

def test_wilcoxon(days):
    # Cargar el CSV
    data= read_data(days ,['open', 'close', 'date'])

    data["returns"]= (data['close']- data['open'])/ data['open']

    # Comparar variable return antes y después de un evento (suponiendo que tienes dos subconjuntos)
    before_event = data['returns'][data['date'] < '2024-12-24']
    after_event = data['returns'][data['date'] >= '2024-12-24']

    # Asegúrate de que ambos conjuntos tengan la misma longitud
    min_length = min(len(before_event), len(after_event))

    # Truncar ambos conjuntos al mínimo común
    before_event = before_event.iloc[:min_length]
    after_event = after_event.iloc[:min_length]

    # Hipótesis
    print("Hipótesis nula (H0): la mediana de la variable return antes del evento es igual a la mediana de la variable return después del evento.")
    print("Hipótesis alternativa (H1): la mediana de la variable return antes del evento no es igual a la mediana de la variable return después del evento.")

    wilcoxon_statistic, wilcoxon_p_value = stats.wilcoxon(before_event, after_event)
    print(f"Prueba de Wilcoxon: estadístico = {wilcoxon_statistic}, valor p = {wilcoxon_p_value}")

    # Interpretación
    if wilcoxon_p_value < 0.05:
        print("Se rechaza la hipótesis nula: la variable return antes y después del evento es significativamente diferente.")
    else:
        print("No hay suficiente evidencia para rechazar la hipótesis nula.")


def test_kruskal_wallis(days):
    # Supongamos que estás comparando la volatilidad de diferentes criptomonedas
    data= read_data(days , ['volatility', 'symbol'])
    group1 = data[data['symbol'] == 'BTC/USD']['volatility']
    group2 = data[data['symbol'] == 'BAT/USD']['volatility']
    group3 = data[data['symbol'] == 'LTC/USD']['volatility']
    
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


def test_spearman( days):
    data= read_data(days , ['volatility', 'open', 'close'])
    data['returns']= (data['close']- data['open']) /data['open']

    # Hipótesis
    print("Correlación de Spearman")
    print("Hipótesis nula (H0): no hay correlación entre la variable returns y volatilidad.")
    print("Hipótesis alternativa (H1): hay una correlación entre la variable returns y volatilidad.")

    # Evaluar la correlación entre el la variable returns y la volatilidad
    spearman_corr, spearman_p_value = stats.spearmanr(data['returns'], data['volatility'])
    print(f"Correlación de Spearman: coeficiente = {spearman_corr}, valor p = {spearman_p_value}")

    # Interpretación
    if spearman_p_value < 0.05:
        print("Hay una correlación significativa entre el la variable returns y la volatilidad.")
    else:
        print("No hay suficiente evidencia para concluir que existe una correlación.")      


def test_mannwhitneyu(days):
    data= read_data(days, ['open', 'close', 'symbol'])

    data['returns']= (data['close']- data['open']) /data['open']
    # Dividir los datos en dos grupos, por ejemplo, comparar la volatilidad de dos criptomonedas
    group1 = data[data['symbol'] == 'BTC/USD']['returns']
    group2 = data[data['symbol'] == 'ETH/USD']['returns']

    # Hipótesis
    print("Prueba de Mann-Whitney U")
    print("Hipótesis nula (H0): las distribuciones de la Variable Returns de BTC y ETH son iguales.")
    print("Hipótesis alternativa (H1): las distribuciones de la Variable Returns de BTC y ETH son diferentes.")

    # Realizar la prueba de Mann-Whitney U
    mannwhitney_statistic, mannwhitney_p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    print(f"Prueba de Mann-Whitney U: estadístico = {mannwhitney_statistic}, valor p = {mannwhitney_p_value}")

    # Interpretación
    if mannwhitney_p_value < 0.05:
        print("Se rechaza la hipótesis nula: hay una diferencia significativa en la Variable Returns entre BTC y ETH.")
    else:
        print("No hay suficiente evidencia para rechazar la hipótesis nula.")

    # Gráfica de las distribuciones
    plt.figure(figsize=(12, 6))

    # Histograma de las distribuciones
    sns.histplot(group1, bins=30, color='blue', label='BTC', stat='density', kde=True, alpha=0.6)
    sns.histplot(group2, bins=30, color='orange', label='ETH', stat='density', kde=True, alpha=0.6)

    plt.title('Distribuciones de Variable Returns de BTC y ETH')
    plt.xlabel('Variable Returns')
    plt.ylabel('Densidad')
    plt.legend()
    plt.grid()
    plt.show()    
   