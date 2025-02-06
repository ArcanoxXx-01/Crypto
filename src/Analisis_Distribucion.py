import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from read_data import read_data
import warnings
# Suprimir warnings
warnings.filterwarnings("ignore")



def show_dist(var, days):

    data = read_data(days, [var])

    # Suponiendo que 'data_sample' es tu conjunto de datos
    data_sample = data[var]  # Puedes reemplazarlo con tus datos

    # Definición de las distribuciones a probar
    distributions = {
        'norm': stats.norm,
        'lognorm': stats.lognorm,
        'expon': stats.expon,
        'pareto': stats.pareto,
        'beta': stats.beta,
        'Weibull': stats.weibull_max,
        'Logística': stats.logistic,
        'Gumbel': stats.gumbel_r,
        'T de Student': stats.t,
    }

    # Configuración del gráfico
    fig, ax = plt.subplots(len(distributions), 2, figsize=(12, 3 * len(distributions)))

    # Ajuste y gráfico para cada distribución
    for i, (name, distribution) in enumerate(distributions.items()):
        # Ajustar la distribución a los datos
        params = distribution.fit(data_sample)

        print(params)

        # Crear el gráfico PDF de la distribución ajustada
        x = np.linspace(min(data_sample), max(data_sample), 1000)
        pdf_fitted = distribution.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])

        # Estimación de la densidad empírica
        kde = stats.gaussian_kde(data_sample)
        empirical_pdf = kde(x)

        # Graficar histogramas y PDFs
        ax[i, 0].hist(data_sample, bins=30, density=True, alpha=0.6, color='g', label='Datos')
        ax[i, 0].plot(x, empirical_pdf, 'b-', lw=2, label='Densidad empírica (KDE)')
        ax[i, 0].plot(x, pdf_fitted, 'r-', lw=2, label=f'{name} ajustada')
        ax[i, 0].set_title(f'Distribución {name}')
        ax[i, 0].legend()

        # Gráfico Q-Q (cuantiles) para ver ajuste
        stats.probplot(data_sample, dist=distribution, sparams=params, plot=ax[i, 1])
        ax[i, 1].set_title(f'Q-Q plot para {name}')

    plt.tight_layout()
    plt.show()


def test_de_bondad(var, days):
    
    data = read_data(days, [var])
    # Definición de las distribuciones a probar
    distributions = {
        'weibull_max': stats.weibull_max,
        'gumbel_r': stats.gumbel_r,
        'norm': stats.norm,
        'lognorm': stats.lognorm,
        'expon': stats.expon,
        'pareto': stats.pareto,
        'beta': stats.beta,
        'logistic': stats.logistic,
        't': stats.t,
    }

    for name, distribution in distributions.items():
        # Ajustar la distribución a los datos
        params = distribution.fit(data[var])
        
        # Prueba de Kolmogorov-Smirnov
        ks_statistic, ks_p_value = stats.kstest(data[var], name, args=params)
        
        # Imprimir resultados de la prueba
        print(f"Prueba K-S para {name}: estadístico = {ks_statistic}, valor p = {ks_p_value}")

        # Conclusiones sobre la hipótesis nula
        if ks_p_value < 0.05:
            print(f"Los datos no siguen la distribución {name} (K-S test).")
        else:
            print(f"Los datos podrían seguir la distribución {name} (K-S test).")
        print()  # Espacio adicional para claridad

        



