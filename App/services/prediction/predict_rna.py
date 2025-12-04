import joblib
import numpy as np
from keras.models import load_model

# Columnas REALES usadas en el entrenamiento de la RNA
# Columnas EXACTAS en el mismo orden que FEATURES_RNA
COLUMNAS_RNA = [
    "IdProducto",
    "IdMarca",
    "IdPresentacion",
    "VM",
    "PM3",
    "T",
    "Prediccion",
    "Me",
    "A"
]


def recomendar(df_rna):

    # Cargar scaler y modelo
    scaler = joblib.load("App/models/rna_scaler.pkl")
    model = load_model("App/models/rna_model.keras")

    # Seleccionar features de entrada
    X = df_rna[COLUMNAS_RNA].values

    # Escalar datos
    X_scaled = scaler.transform(X)

    # Predicción de clases
    pred_probs = model.predict(X_scaled)
    pred_class = np.argmax(pred_probs, axis=1)

    # Copia + columna final
    df_out = df_rna.copy()
    df_out["Accion_Recomendada"] = pred_class

    return df_out
