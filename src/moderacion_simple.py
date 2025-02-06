import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from read_data import read_data
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

    print(model.summary())

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


def high_volume_open(days, symbol= None):
 # Definición de las variables
    data = read_data(days, ['high', 'volume cripto', 'open', 'symbol'])
    
    if symbol:
        data= data[data['symbol']==symbol]
    # Agregar una variable de interacción (moderadora)
    data['interaction'] = data['volume cripto'] * data['open']

    # Definir variables dependientes e independientes
    X = data[['volume cripto', 'open', 'interaction']]
    y = data['high']

    # Añadir una constante al modelo
    X = sm.add_constant(X)

    # Ajuste del modelo de regresión
    model = sm.OLS(y, X).fit()

    # Mostrar resumen de los resultados
    print(model.summary())

    # Gráfica de los resultados
    plt.figure(figsize=(10, 6))
    # Gráfico de dispersión para la relación entre returns y volatility
    sns.scatterplot(x='volume cripto', y='high', data=data, alpha=0.5)

    # Graficar la línea de regresión para diferentes niveles de la variable moderadora 'open'
    open_values = [data['open'].min(), data['open'].mean(), data['open'].max()]
    for o in open_values:
        # Obtener las predicciones de la línea de regresión
        interaction_term = data['volume cripto'] * o
        predictions = model.predict(sm.add_constant(data[['volume cripto', 'open']].assign(interaction=interaction_term)))

        # Graficar la línea de regresión
        plt.plot(data['volume cripto'], predictions, label=f'Open = {o:.2f}', linestyle='--')

    plt.title('Moderación de Open en la Relación entre high y volume cripto')
    plt.xlabel('volume cripto')
    plt.ylabel('high')
    plt.legend(title='Niveles de Open')
    plt.show()