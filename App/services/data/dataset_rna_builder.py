import pandas as pd

def construir_dataset_rna(df_pls):
    """
    Construye dataset para la RNA basado en datos del PLS.
    Usa reglas mejoradas y más naturales:
        - ratio = Pred / PM3
        - T (tendencia)
        - PM3 como referencia histórica
        - Prediccion como expectativa futura
    """

    df = df_pls.copy()
    acciones = []

    for _, row in df.iterrows():
        pred = row["Prediccion"]
        pm3 = row["PM3"]
        t = row["T"]
        vm = row["VM"]

        # Evitar división por cero
        if pm3 <= 0:
            acciones.append(1)
            continue

        ratio = vm / pm3      # ahora ratio compara REALIDAD vs HISTORIAL
        ratio_pred = pred / pm3

        # ---------------------------------------------------------
        # 🔵 ACCIÓN 3 — AUMENTO FUERTE (alta demanda sostenida)
        # ---------------------------------------------------------
        if ratio > 1.30 and t > 10 and ratio_pred > 1.10:
            accion = 3

        # ---------------------------------------------------------
        # 🟢 ACCIÓN 2 — AUMENTO MODERADO (crecimiento suave)
        # ---------------------------------------------------------
        elif ratio > 1.10 and t > 0:
            accion = 2

        # ---------------------------------------------------------
        # 🟡 ACCIÓN 1 — MANTENER (estabilidad)
        # ---------------------------------------------------------
        elif 0.90 <= ratio <= 1.10 and abs(t) < 15:
            accion = 1

        # ---------------------------------------------------------
        # 🔴 ACCIÓN 0 — REDUCIR (ventas cayendo)
        # ---------------------------------------------------------
        elif ratio < 0.90 and (t < 0 or pred < pm3):
            accion = 0

        # ---------------------------------------------------------
        # Caso por defecto: mantener
        # ---------------------------------------------------------
        else:
            accion = 1

        acciones.append(accion)

    df["Accion"] = acciones

    columnas = [
        "IdProducto", "IdMarca", "IdPresentacion",
        "VM", "PM3", "T",
        "Prediccion",
        "Me", "A",
        "Accion"
    ]

    df_rna = df[columnas].copy()

    df_rna.to_csv("data/datasetRNA.csv", sep=";", index=False)

    return df_rna
