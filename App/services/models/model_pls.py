import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression

def entrenar_pls(df):
    # Columnas categóricas y numéricas
    cat_cols = ['IdProducto','IdMarca','IdPresentacion','IdCliente']
    num_cols = ['VM','PM3','T','Me','A']

    X = df[cat_cols + num_cols]
    y = df['Cantidad']

    # Preprocesamiento
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
    ])

    # Pipeline con preprocesamiento + PLS
    pls_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('pls', PLSRegression(n_components=2))
    ])

    # Entrenar
    pls_pipeline.fit(X, y)

    # Guardar modelo
    joblib.dump(pls_pipeline, "App/models/pls_model.pkl")
    print("Modelo PLS entrenado y guardado en App/models/pls_model.pkl")

    return pls_pipeline





'''
def entrenar_pls(df):
    X = df[['VM','PM3','T','IdProducto','IdMarca','IdPresentacion','IdCliente','Me','A']]
    y = df['Cantidad']

    pls = PLSRegression(n_components=2)
    pls.fit(X, y)

    joblib.dump(pls, "App/models/pls_model.pkl")
    return pls
'''
