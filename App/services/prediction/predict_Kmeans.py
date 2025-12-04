
import joblib
from sklearn.metrics import silhouette_score, davies_bouldin_score

def predecir_kmeans(df):
    kmeans = joblib.load("App/models/kmeans_model.pkl")
    scaler = joblib.load("App/models/kmeans_scaler.pkl")

    X = df[["TC", "F", "PC", "PD", "UC"]]
    X_scaled = scaler.transform(X)

    df["Cluster"] = kmeans.predict(X_scaled)
    return df


def evaluar_kmeans(df):
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    
    X = df[["TC", "F", "PC", "PD", "UC"]]
    kmeans = joblib.load("App/models/kmeans_model.pkl")
    scaler = joblib.load("App/models/kmeans_scaler.pkl")

    X_scaled = scaler.transform(X)
    labels = kmeans.predict(X_scaled)

    # Silhouette Score
    silhouette = silhouette_score(X_scaled, labels)

    # Davies-Bouldin Index
    dbi = davies_bouldin_score(X_scaled, labels)

    # Inertia (SSE interno)
    inertia = kmeans.inertia_

    return silhouette, dbi, inertia
