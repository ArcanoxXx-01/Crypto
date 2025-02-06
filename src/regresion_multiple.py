import pandas as pd
import statsmodels.api as sm
from read_data import read_data

def multiple_regression(days: int, independient_vars, dependient_var, type: None):
    # Cargar los datos
    vars= independient_vars
    vars.append(dependient_var)
    df = read_data(days, vars+['symbol'] )

    # Eliminar filas con valores nulos
    df = df.dropna(subset=vars)

    if type: 
        df= df[df['symbol']==type]

    # Asegurar que las columnas son del tipo correcto
    numeric_columns = vars
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Variables independientes y dependiente
    X = df[independient_vars]
    y = df[dependient_var]

    # Agregar constante para el modelo
    X = sm.add_constant(X)

    # Ajustar el modelo de regresión múltiple
    try:
        model = sm.OLS(y, X).fit()
        print(model.summary())
    except Exception as e:
        print(f"Error al ajustar el modelo: {e}")

