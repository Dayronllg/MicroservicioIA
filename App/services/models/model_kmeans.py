import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def entrenar_kmeans(df, k=2):
    X = df[["TC", "F", "PC", "PD", "UC"]]

    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entrenar KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)

    # Guardar modelo
    joblib.dump(kmeans, "App/models/kmeans_model.pkl")
    joblib.dump(scaler, "App/models/kmeans_scaler.pkl")

    return kmeans, scaler
