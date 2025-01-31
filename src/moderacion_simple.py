import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from read_data import read_data
import matplotlib.pyplot as plt
import numpy as np

def moderacion_simple_volume_cripto_usd_average(days):
    # Cargar los datos necesarios
    data = read_data(days, ['volume cripto', 'volume usd', 'average'])
    
    # Estandarización de las variables (opcional, pero ayuda con la interpretación)
    data['volume_crypto_std'] = (data['volume cripto'] - data['volume cripto'].mean()) / data['volume cripto'].std()
    data['average_std'] = (data['average'] - data['average'].mean()) / data['average'].std()

    # Crear la interacción entre 'volume cripto' y 'average'
    data['interaction'] = data['volume_crypto_std'] * data['average_std']

    # Modelo de regresión incluyendo la interacción
    formula = 'Q("volume usd") ~ volume_crypto_std + average_std + interaction'
    model = smf.ols(formula=formula, data=data).fit()

    # Resumen del modelo
    print(model.summary())

    # Crear grupos de 'average' (bajo, medio, alto)
    data['average_group'] = pd.qcut(data['average'], q=3, labels=['Bajo', 'Medio', 'Alto'])

    # Gráfica de interacción
    plt.figure(figsize=(10, 6))
    for group in data['average_group'].unique():
        subset = data[data['average_group'] == group]
        plt.scatter(subset['volume cripto'], subset['volume usd'], label=f'Average: {group}', alpha=0.6)

    plt.title('Interacción entre Volume Cripto y Average en Relación a Volume USD')
    plt.xlabel('Volume Cripto')
    plt.ylabel('Volume USD')
    plt.legend()
    plt.grid()
    plt.show()


def moderacion_open_close_volatility(days):
    df= read_data(days, ['volatility', 'open', 'close'])
    # Crear la interacción entre open y volatility
    df['open_volatility_interaction'] = df['open'] * df['volatility']

    # Ajustar el modelo de moderación
    model = smf.ols(formula='close ~ open + volatility + open_volatility_interaction', data=df).fit()

    # Predicciones para diferentes niveles de volatilidad
    volatility_levels = [df['volatility'].quantile(0.25),  # Baja volatilidad (percentil 25)
                        df['volatility'].median(),        # Media volatilidad
                        df['volatility'].quantile(0.75)]  # Alta volatilidad (percentil 75)

    # Crear un DataFrame para las predicciones
    pred_data = pd.DataFrame({
        'open': np.linspace(df['open'].min(), df['open'].max(), 100),  # Valores de apertura
    })

    # Añadir predicciones para cada nivel de volatilidad
    plt.figure(figsize=(10, 6))
    for v in volatility_levels:
        pred_data['volatility'] = v
        pred_data['open_volatility_interaction'] = pred_data['open'] * v
        pred_data['predicted_close'] = model.predict(pred_data)
        plt.plot(pred_data['open'], pred_data['predicted_close'], label=f'Volatility: {v:.2f}')

    # Configuración del gráfico
    plt.scatter(df['open'], df['close'], alpha=0.3, label="Datos reales")
    plt.title("Moderación de la relación Open-Close por la volatilidad")
    plt.xlabel("Precio de apertura (Open)")
    plt.ylabel("Precio de cierre (Close)")
    plt.legend(title="Niveles de Volatility")
    plt.grid(True)
    plt.show()
