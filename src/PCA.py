import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from read_data import read_data

def pca (days):
    features = ['open', 'high', 'low', 'close', 'volume usd', 'volatility']

    data_subset = read_data(days, features)

    # Estandarizar los datos
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_subset)

    # Aplicar PCA
    pca = PCA(n_components= 2)  # Elegimos dos componentes para visualización
    principal_components = pca.fit_transform(data_scaled)

    # Convertir los resultados a un DataFrame
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    # Gráfico de dispersión de los componentes principales
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
    plt.title('PCA: Componentes Principales')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.grid()
    plt.show()

    # Explicación de la varianza
    explained_variance = pca.explained_variance_ratio_
    print(f"Varianza explicada por los componentes: {explained_variance}")