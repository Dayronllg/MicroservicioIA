import pandas as pd

import pandas as pd

def crear_variables(df):
    # Última venta del mes anterior por producto
    df['VM'] = df.groupby('IdProducto')['Cantidad'].shift(1)
    
    # Promedio móvil últimos 3 meses por producto
    df['PM3'] = df.groupby('IdProducto')['Cantidad'].rolling(3).mean().shift(1).reset_index(0, drop=True)
    
    # Tendencia histórica (diferencia)
    df['T'] = df.groupby('IdProducto')['Cantidad'].transform(lambda x: x.diff().fillna(0))
    
    # Mes y año
    df['Me'] = pd.to_datetime(df['Fecha']).dt.month
    df['A']  = pd.to_datetime(df['Fecha']).dt.year
    
    # Eliminamos filas que tengan NaN en VM o PM3
    return df.dropna(subset=['VM', 'PM3'])

