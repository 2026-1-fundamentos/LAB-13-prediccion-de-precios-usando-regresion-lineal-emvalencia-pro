import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Paso 1: Preprocesamiento
def cargar_y_limpiar(ruta):
    df = pd.read_csv(ruta, index_col=False, compression="zip")
    df["Age"] = 2021 - df["Year"]
    df.drop(columns=["Year", "Car_Name"], inplace=True)
    return df


# Paso 2: División en X e y
def separar_features_target(df):
    x = df.drop(columns=["Present_Price"])
    y = df["Present_Price"]
    return x, y


# Paso 3: Pipeline
def construir_pipeline():
    categoricas = ["Fuel_Type", "Selling_type", "Transmission", "Owner"]
    numericas = ["Selling_Price", "Driven_kms", "Age"]

    transformador = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(handle_unknown="ignore"), categoricas),
            ("scaler", MinMaxScaler(), numericas),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocesamiento", transformador),
            ("SelectKBest", SelectKBest(score_func=f_regression)),
            ("modelo", LinearRegression()),
        ]
    )
    return pipeline


# Paso 4: Optimización de hiperparámetros
def optimizar(x_train, y_train):
    param_grid = {
        "SelectKBest__k": list(range(1, 20)),
    }

    busqueda = GridSearchCV(
        estimator=construir_pipeline(),
        param_grid=param_grid,
        cv=10,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=2,
    )

    busqueda.fit(x_train, y_train)
    return busqueda


# Paso 5: Guardar modelo
def guardar_modelo(modelo):
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(modelo, f)


# Paso 6: Métricas
def calcular_metricas(nombre, y_real, y_pred):
    return {
        "type": "metrics",
        "dataset": nombre,
        "r2": round(r2_score(y_real, y_pred), 4),
        "mse": round(mean_squared_error(y_real, y_pred), 4),
        "mad": round(mean_absolute_error(y_real, y_pred), 4),
    }


def guardar_metricas(lista_metricas):
    os.makedirs("files/output", exist_ok=True)
    with open("files/output/metrics.json", "w", encoding="utf-8") as f:
        for m in lista_metricas:
            f.write(json.dumps(m) + "\n")


def main():
    train = cargar_y_limpiar("files/input/train_data.csv.zip")
    test = cargar_y_limpiar("files/input/test_data.csv.zip")

    x_train, y_train = separar_features_target(train)
    x_test, y_test = separar_features_target(test)

    modelo = optimizar(x_train, y_train)

    guardar_modelo(modelo)

    pred_train = modelo.predict(x_train)
    pred_test = modelo.predict(x_test)

    metricas = [
        calcular_metricas("train", y_train, pred_train),
        calcular_metricas("test", y_test, pred_test),
    ]

    guardar_metricas(metricas)


if __name__ == "__main__":
    main()
