import pandas as pd

def cargar_datos(ruta="data/datasetDistribuidora.csv"):
    df = pd.read_csv(ruta, sep=";")
    return df




def cargar_datos_kmeans():
    try:
        df = pd.read_csv("data/datasetDistribuidoraKmeans.csv", sep=";", encoding="utf-8-sig")
    except:
        df = pd.read_csv("data/datasetDistribuidoraKmeans.csv", 
                         sep=";", 
                         header=None,
                         names=["IdCliente", "TipoCliente", "IdVenta", "IdProducto", "Subtotal", "Fecha"],
                         encoding="utf-8-sig")

    # Normalizar nombres
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace("\ufeff", "")  # limpia BOM al inicio

    return df

