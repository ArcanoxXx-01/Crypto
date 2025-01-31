import os
import pandas as pd # type: ignore


# Configuración: directorio donde están los archivos CSV
DIRECTORIO_CSV = "./db"
COLUMNAS_A_ELIMINAR = ['unix']

def repair( delete_first_row = 0 ):
    """
    Procesa todos los archivos CSV en el directorio configurado ( /db ):
    1. Elimina las columnas 'unix' y 'symbol'  (si existen).
    2. Cambia el nombre de la última columna a 'Volume Cripto' (si no existe).
    3. Agrega una nueva columna 'volatility', calculada como (high - low) / open.  (si no existe).
    4. Agrega una nueva columna 'average' , calculada como Volume USD / Volume Cripto.  (si no existe).
    """
    if not os.path.isdir(DIRECTORIO_CSV):
        print(f"El directorio '{DIRECTORIO_CSV}' no existe. Por favor, crea un directorio llamado 'db' y agrega los archivos CSV.")
        return

    for archivo in os.listdir(DIRECTORIO_CSV):
        if archivo.endswith('.csv'):
            ruta_archivo = os.path.join(DIRECTORIO_CSV, archivo)
            try:
                # Leer el archivo CSV
                df = pd.read_csv(ruta_archivo, skiprows = 1)
                 # Normalizar los nombres de las columnas
                df.columns = df.columns.str.strip().str.lower()
                # Eliminar las columnas especificadas si existen
                df = df.drop(columns=[col for col in COLUMNAS_A_ELIMINAR if col in df.columns], errors='ignore')
                
                # Cambiar el nombre de la última columna a "Volume Cripto"
                if len(df.columns) > 0 and not {'volume cripto'}.issubset(df.columns):  # Verificar que haya columnas  
                    ultima_columna = df.columns[-1]
                    df.rename(columns={ultima_columna: "volume cripto"}, inplace=True)
                
                # Agregar la nueva columna 'volatility'
                if {'high', 'low', 'open'}.issubset(df.columns) and not {'volatility'}.issubset(df.columns):
                    df.insert(5,"volatility",(df['high'] - df['low']) / df['open'])

                # Agregar la nueva columna 'average'
                if {'volume usd', 'volume cripto'}.issubset(df.columns) and not {'average'}.issubset(df.columns):
                    df.insert(6,"average",df['volume usd'] / df['volume cripto'])

                # Guardar el archivo actualizado
                df.to_csv(ruta_archivo, index=False)
                print(f"Procesado: {archivo}")
            except Exception as e:
                print(f"Error procesando {archivo}: {e}")

# # Llamar automáticamente a la función al importar el script
# if __name__ == "__main__":
repair()