
from read_data import read_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


def pca(days):
    # Seleccionar las características
    features = ['open', 'high', 'low', 'close', 'volume cripto', 'average', 'volatility', 'returns']

    df = read_data(days, features+['symbol'])
    df['symbol'] = df['symbol'].str.strip()

    X = df[features]

    # Estandarizar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Aplicar PCA
    n_components = 2 # Puedes ajustar este valor
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)

    # Varianza explicada
    explained_variance = pca.explained_variance_ratio_
    print(f"Varianza explicada por los componentes: {explained_variance}")

    # Cargas de las variables
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(loadings, columns=[f'PC{i+1}' for i in range(n_components)], index=features)
    print("\nCargas de las variables:")
    print(loadings_df)

    # Gráfico de las componentes principales
    plt.figure(figsize=(8, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Componentes Principales')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance)
    plt.xlabel("Componente Principal")
    plt.ylabel("Proporción de Varianza Explicada")
    plt.title("Varianza Explicada por Componente Principal")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.heatmap(loadings_df, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Heatmap de Cargas de Componentes Principales')
    plt.show()



def analisis_pca_criptos(days):
    # Seleccionar las características
    features = ['open', 'high', 'low', 'close', 'volume cripto', 'average', 'volatility','returns']

    df = read_data(days, features+['symbol'])
    df['symbol'] = df['symbol'].str.strip()  

    # --- 2. PCA por Criptomoneda ---
    plt.figure(figsize=(12, 8))

    for symbol in df['symbol'].unique():
        cripto_data = df[df['symbol'] == symbol]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cripto_data[features])
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X_scaled)
        
        explained_variance = pca.explained_variance_ratio_
        print(f"\nResultados para {symbol}:")
        print(f"Varianza explicada: {explained_variance}")

        # Cargas de las variables
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loadings_df = pd.DataFrame(loadings, columns=[f'PC{i+1}' for i in range(2)], index=features)
        print("\nCargas de las variables:")
        print(loadings_df)
        
        # Obtener la media de los componentes principales para cada criptomoneda
        mean_pc1 = np.mean(principal_components[:, 0])
        mean_pc2 = np.mean(principal_components[:, 1])
        
        plt.scatter(mean_pc1, mean_pc2, label=symbol, s=100) # s=100 para puntos más grandes

    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Representación de Criptomonedas en el Espacio de Componentes Principales (PCA por Criptomoneda)')
    plt.legend()
    plt.show()   

    plt.figure(figsize=(12, 8))

    for symbol in df['symbol'].unique():
        cripto_data = df[df['symbol'] == symbol]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cripto_data[features])
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X_scaled)

        plt.plot(principal_components[:, 0], principal_components[:, 1], label=symbol, marker='o', linestyle='-', alpha=0.7)

    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Trayectorias de Criptomonedas en el Espacio de Componentes Principales')
    plt.legend()
    plt.show()  

