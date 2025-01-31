import pandas as pd
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from read_data import read_data


def AFE():
    # Seleccionar las variables relevantes para el AFE
    variables = ['open', 'high', 'low', 'close', 'average', 'volume usd', 'volatility', 'volume cripto']

    # Cargar el dataset (reemplaza con tu archivo CSV)
    data = read_data(100, variables)

    # Estandarizar las variables
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Prueba de adecuación de KMO
    from factor_analyzer.factor_analyzer import calculate_kmo
    kmo_all, kmo_model = calculate_kmo(data_scaled)
    print(f"KMO de adecuación: {kmo_model:.2f}")
    if kmo_model < 0.6:
        print("Los datos no son adecuados para AFE.")
    else:
        print("Los datos son adecuados para AFE.")

    # Prueba de esfericidad de Bartlett
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
    chi2, p = calculate_bartlett_sphericity(data_scaled)
    print(f"Prueba de Bartlett: chi2={chi2:.2f}, p={p:.2e}")
    if p < 0.05:
        print("Las variables están correlacionadas y son adecuadas para AFE.")
    else:
        print("Las variables no están correlacionadas, considera revisar los datos.")

    # Crear el modelo de AFE
    fa = FactorAnalyzer(n_factors=2, rotation="varimax")  # Ajusta el número de factores
    fa.fit(data_scaled)

    # Valores propios (eigenvalues)
    eigenvalues, _ = fa.get_eigenvalues()
    print("Valores propios:", eigenvalues)

    # Gráfico de sedimentación (scree plot)
    plt.plot(range(1, len(eigenvalues)+1), eigenvalues, marker='o')
    plt.title('Gráfico de sedimentación (Scree Plot)')
    plt.xlabel('Factor')
    plt.ylabel('Valor propio')
    plt.grid()
    plt.show()

    # Cargas factoriales
    factor_loadings = pd.DataFrame(fa.loadings_, index=variables, columns=[f"Factor {i+1}" for i in range(2)])
    print("Cargas factoriales:")
    print(factor_loadings)

    # Interpretación de los factores
    print("Varianza explicada por factor:")
    variance = fa.get_factor_variance()
    print(f"Varianza explicada: {variance[0]}")


AFE()