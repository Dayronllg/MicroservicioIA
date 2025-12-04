import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from services.data.data_loader import cargar_datos, cargar_datos_kmeans
from services.data.feature_engineering import crear_variables
from services.data.feature_engineering_Kmeans import crear_variables_kmeans
from services.data.dataset_rna_builder import construir_dataset_rna

from services.models.model_pls import entrenar_pls
from services.models.model_kmeans import entrenar_kmeans
from services.models.model_rna import entrenar_rna

from services.prediction.predict_pls import predecir, evaluar
from services.prediction.predict_Kmeans import predecir_kmeans, evaluar_kmeans
from services.prediction.predict_rna import recomendar

from sklearn.model_selection import train_test_split


def main():

    print("=== ENTRENANDO MODELO PLS (Pronóstico de ventas) ===")

    # 1. Cargar y preparar datos
    df = cargar_datos()
    df = crear_variables(df)

    # 2. Dividir para ENTRENAR el PLS (solo para entrenarlo)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # 3. Entrenar PLS SOLO con datos de entrenamiento
    entrenar_pls(df_train)

    # 4. Predecir TODO el dataset (NO solo el test)
    df = predecir(df)

    # 5. Evaluación usando SOLO test (como debe ser)
    mae, mse, rmse, r2 = evaluar(df_test.assign(Prediccion=df.loc[df_test.index, "Prediccion"]))

    print("\nResultados PLS:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")

    # ------------------------------------------------------------------
    # 🔹 MODULO KMEANS (completamente independiente)
    # ------------------------------------------------------------------

    print("\n=== ENTRENANDO KMEANS (Segmentación clientes) ===")

    df_km = cargar_datos_kmeans()
    df_km = crear_variables_kmeans(df_km)

    entrenar_kmeans(df_km)
    df_km = predecir_kmeans(df_km)

    sil, dbi, inertia = evaluar_kmeans(df_km)

    print("\nMétricas KMeans:")
    print(f"Silhouette Score: {sil:.4f}")
    print(f"Davies-Bouldin Index: {dbi:.4f}")
    print(f"Inertia: {inertia:.2f}")

    # ------------------------------------------------------------------
# 🔹 RNA (usa SOLO el dataset del PLS, NO usa Kmeans)
# ------------------------------------------------------------------

    print("\n=== ENTRENANDO RNA (Sistema de recomendación) ===")

    # 6. Construir dataset REAL de la RNA
    df_rna = construir_dataset_rna(df)

    print(f"Registros para RNA: {len(df_rna)}")

    accuracy = entrenar_rna(df_rna)

    print(f"RNA Accuracy: {accuracy:.4f}")

# 7. Hacer recomendaciones
    print("\n=== RECOMENDACIONES RNA ===")
    df_out = recomendar(df_rna)
    print(df_out.head())

    print("\n=== 20 RECOMENDACIONES RNA ===")

# Ordenar por IdProducto, Mes, Año para que tenga sentido
    df_show = df_out.sort_values(by=["IdProducto", "A", "Me"]).head(600)

    print(df_show.to_string())


if __name__ == "__main__":
    main()

