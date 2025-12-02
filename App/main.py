from services.data.data_loader import cargar_datos
from services.data.feature_engineering import crear_variables
from services.models.model_pls import entrenar_pls
from services.prediction.predict_pls import predecir
from services.prediction.predict_pls import evaluar
from sklearn.model_selection import train_test_split

def main():
    df = cargar_datos()

    df = crear_variables(df)

    modelo = entrenar_pls(df)

    print("Modelo entrenado y guardado como PKL")

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # Entrenar modelo con train
    modelo = entrenar_pls(df_train)
    print("Modelo entrenado y guardado como PKL")

    # Predecir en test
    df_test = predecir(df_test)
   
    print(df_test[['Cantidad', 'Prediccion']].head(10))

    # Evaluar métricas
    mae, mse, rmse, r2 = evaluar(df_test)
    print("\nMétricas de efectividad en Test:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")

if __name__ == "__main__":
    main()

