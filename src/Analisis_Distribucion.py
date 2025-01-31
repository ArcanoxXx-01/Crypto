import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from read_data import read_data


data = read_data(50, ['volatility', 'open', 'close'])
data['returns']= (data['close']- data['open'])/ data['open']

def show_dist(var):
    
    plt.figure(figsize=(12, 6))
    sns.histplot(data[var], bins=30, kde=True)
    plt.title(f'Distribución de {var}')
    plt.xlabel(var)
    plt.ylabel('Frecuencia')
    plt.show()

    # Ajustando varias distribuciones
    distributions = {
        'norm': stats.norm,
        'lognorm': stats.lognorm,
        'expon': stats.expon,
        'pareto': stats.pareto,
        'beta': stats.beta,
        'gamma': stats.gamma,
        'Weibull': stats.weibull_max,
        'Logística': stats.logistic,
        'Gumbel': stats.gumbel_r,
        'T de Student': stats.t,
    }

    # Evaluar la bondad de ajuste
    for name, distribution in distributions.items():
        param = distribution.fit(data[var])
        
        # Crear los datos ajustados 
        x = np.linspace(min(data[var]), max(data[var]), 100)
        pdf_fitted = distribution.pdf(x, *param)

        # Graficar la distribución ajustada
        plt.figure(figsize=(10, 6))
        sns.histplot(data[var], bins=30, stat='density', label='Datos Observados', color='blue', kde=True)
        plt.plot(x, pdf_fitted, label=f'Distribución Ajustada: {name}', color='red')
        plt.title(f'Comparación de Distribución Ajustada - {name}')
        plt.xlabel(var)
        plt.ylabel('Densidad')
        plt.legend()
        plt.show()


def test_de_bondad():
    # Ajustar la distribución Weibull
    params_weibull = stats.weibull_max.fit(data)

    # Ajustar la distribución Gumbel
    params_gumbel = stats.gumbel_r.fit(data)
    # Prueba de Kolmogorov-Smirnov para Weibull
    ks_statistic_weibull, ks_p_value_weibull = stats.kstest(data, 'weibull_max', args=params_weibull)
    print(f"Prueba K-S para Weibull: estadístico = {ks_statistic_weibull}, valor p = {ks_p_value_weibull}")

    # Conclusiones para Weibull
    if ks_p_value_weibull < 0.05:
        print("Los datos no siguen la distribución Weibull (K-S test).")
    else:
        print("Los datos podrian seguir la distribución Weibull (K-S test).")

    # Prueba de Kolmogorov-Smirnov para Gumbel
    ks_statistic_gumbel, ks_p_value_gumbel = stats.kstest(data, 'gumbel_r', args=params_gumbel)
    print(f"Prueba K-S para Gumbel: estadístico = {ks_statistic_gumbel}, valor p = {ks_p_value_gumbel}")

    # Conclusiones para Gumbel
    if ks_p_value_gumbel < 0.05:
        print("Los datos no siguen la distribución Gumbel (K-S test).")
    else:
        print("Los datos podrian la distribución Gumbel (K-S test).")


def test_student(var):
    # Ajustar la distribución t-Student
    params_t = stats.t.fit(data['volatility'])

    # Prueba de Kolmogorov-Smirnov para t-Student
    ks_statistic_t, ks_p_value_t = stats.kstest(data['volatility'], 't', args=params_t)
    print(f"Prueba K-S para t-Student: estadístico = {ks_statistic_t}, valor p = {ks_p_value_t}")

    # Conclusiones para t-Student
    if ks_p_value_t < 0.05:
        print("Los datos no siguen la distribución t-Student (K-S test).")
    else:
        print("Los datos podrían seguir la distribución t-Student (K-S test).")
        



