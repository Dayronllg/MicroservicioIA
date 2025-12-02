import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,root_mean_squared_error

def predecir(df):
    modelo = joblib.load("App/models/pls_model.pkl")
    X = df[['VM','PM3','T','IdProducto','IdMarca','IdPresentacion','IdCliente','Me','A']]
    df["Prediccion"] = modelo.predict(X)
    return df

def evaluar(df):
    y_true = df['Cantidad']
    y_pred = df['Prediccion']
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2