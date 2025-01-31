import pandas as pd

df= pd.read_csv('../db/merge/merge.csv')

def read_data(days: int, columns: list):
    # Asegurarte de que la columna de fecha esté en formato datetime
    df['date'] = pd.to_datetime(df['date'])

    # Obtener la fecha más reciente en el DataFrame
    fecha_mas_reciente = df['date'].max()

    # Calcular la fecha límite de hace 3 meses
    fecha_limite = fecha_mas_reciente - pd.DateOffset(days= days)

    # Filtrar las filas con fechas dentro de los últimos 3 meses
    datos_ultimos= df[df['date'] >= fecha_limite]

    # Eliminar valores null 
    values= datos_ultimos[columns].dropna()

    return values

