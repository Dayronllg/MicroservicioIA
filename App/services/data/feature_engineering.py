import pandas as pd

def crear_variables(df):

    # Convertir fecha
    df["Fecha"] = pd.to_datetime(df["Fecha"])

    # 1. Agregamos columnas Año y Mes ANTES de agrupar
    df["A"]  = df["Fecha"].dt.year
    df["Me"] = df["Fecha"].dt.month

    # 2. AGRUPAR A NIVEL MENSUAL por producto/marca/presentación
    df_mensual = (
        df.groupby(["IdProducto", "IdMarca", "IdPresentacion", "A", "Me"])["Cantidad"]
        .sum()
        .reset_index()  # ← aquí ya no crea conflictos
    )

    # 3. Ordenar por producto y tiempo
    df_mensual = df_mensual.sort_values(["IdProducto","A","Me"])

    # 4. Crear variables rezagadas

    # VM = venta del mes anterior
    df_mensual["VM"] = df_mensual.groupby("IdProducto")["Cantidad"].shift(1)

    # PM3 = promedio móvil 3 meses
    df_mensual["PM3"] = (
        df_mensual.groupby("IdProducto")["Cantidad"]
        .rolling(3).mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    # T = tendencia
    df_mensual["T"] = df_mensual.groupby("IdProducto")["Cantidad"].diff()

    # 5. Limpiar filas sin datos históricos
    df_mensual = df_mensual.dropna(subset=["VM", "PM3"])

    return df_mensual
