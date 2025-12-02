import pandas as pd

def cargar_datos(ruta="data/datasetDistribuidora.csv"):
    df = pd.read_csv(ruta, sep=";")
    return df
