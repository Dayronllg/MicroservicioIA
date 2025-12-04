import pandas as pd

def crear_variables_kmeans(df):
    # Convertir fecha
    df["Fecha"] = pd.to_datetime(df["Fecha"])

    # Total Compras (TC)
    TC = df.groupby("IdCliente")["subtotal"].sum()

    # Frecuencia (F)
    F = df.groupby("IdCliente")["IdVenta"].nunique()

    # Promedio Compra (PC)
    PC = TC / F

    # Productos distintos (PD)
    PD = df.groupby("IdCliente")["IdProducto"].nunique()

    # Última compra (UC) en meses
    fecha_actual = df["Fecha"].max()
    UC = (fecha_actual - df.groupby("IdCliente")["Fecha"].max()).dt.days / 30

    # Tipo Cliente (TCli)
    TCli = df.groupby("IdCliente")["TipoCliente"].first()

    # Ensamblar dataset final
    df_final = pd.DataFrame({
        "IdCliente": TC.index,
        "TC": TC.values,
        "F": F.values,
        "PC": PC.values,
        "PD": PD.values,
        "UC": UC.values,
        "TCli": TCli.values
    })

    return df_final

