import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error
)

def predecir(df):
    modelo = joblib.load("App/models/pls_model.pkl")

    # ORDEN CORRECTO (igual al entrenamiento)
    X = df[['IdProducto','IdMarca','IdPresentacion','VM','PM3','T','Me','A']]

    # Predicción del modelo
    pred = modelo.predict(X)

    # CORREGIR PREDICCIONES NEGATIVAS
    pred = np.maximum(pred, 0)

    df["Prediccion"] = pred

    return df


def evaluar(df):
    y_true = df['Cantidad']
    y_pred = df['Prediccion']

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mae, mse, rmse, r2
