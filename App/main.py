from services.data.data_loader import cargar_datos, cargar_datos_kmeans        
from services.data.feature_engineering import crear_variables
from services.data.feature_engineering_Kmeans import crear_variables_kmeans
from services.models.model_pls import entrenar_pls
from services.models.model_kmeans import entrenar_kmeans
from services.prediction.predict_pls import predecir, evaluar
from services.prediction.predict_Kmeans import predecir_kmeans, evaluar_kmeans
from sklearn.model_selection import train_test_split

def main():
    print("=== ENTRENANDO PLS ===")
    df = cargar_datos()
    df = crear_variables(df)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    entrenar_pls(df_train)
    df_test = predecir(df_test)
    mae, mse, rmse, r2 = evaluar(df_test)

    print("PLS:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")

    print("\n=== ENTRENANDO KMEANS ===")

    df_km = cargar_datos_kmeans()
    df_km = crear_variables_kmeans(df_km)

    entrenar_kmeans(df_km)
    df_km = predecir_kmeans(df_km)

    sil, dbi, inertia = evaluar_kmeans(df_km)

    print("Métricas KMeans:")
    print(f"Silhouette Score: {sil:.4f}")
    print(f"Davies-Bouldin Index: {dbi:.4f}")
    print(f"Inertia: {inertia:.2f}")

    print(df_km.head())
    print(df_km.columns)
    print(df_km.head())



if __name__ == "__main__":
    main()
