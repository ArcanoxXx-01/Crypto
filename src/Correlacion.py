import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from read_data import read_data

def correlation_matrix(days: int, columns: list):
    # Cargar el CSV
    data = read_data(days, columns)

    # Asegúrate de eliminar valores nulos
    data = data.dropna()

    # Calcular la matriz de correlación de Spearman
    correlation_spearman = data.corr(method='spearman')

    # Visualizar la matriz de correlación
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_spearman, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Matriz de Correlación de Spearman')
    plt.show()



