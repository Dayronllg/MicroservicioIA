import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

FEATURES_RNA = [
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

def entrenar_rna(df_rna):

    # Variables de entrada
    X = df_rna[FEATURES_RNA].values

    # Etiqueta
    y = df_rna["Accion"].values

    # Número real de clases
    num_classes = int(np.max(y)) + 1
    y_cat = to_categorical(y, num_classes=num_classes)

    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_cat, test_size=0.2, random_state=42
    )

    # RNA
    model = Sequential([
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(X_train, y_train, epochs=40, batch_size=16, verbose=1)

    # Evaluación
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"RNA Accuracy: {accuracy:.4f}")

    # Guardar modelo
    model.save("App/models/rna_model.keras")
    joblib.dump(scaler, "App/models/rna_scaler.pkl")

    return accuracy



