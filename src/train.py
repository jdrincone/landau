#!/usr/bin/env python3
import pandas as pd
import numpy as np
import json
import joblib
import yaml
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def main():
    # Cargar parámetros
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    cv_folds = params["train"].get("cv", 5)
    random_state = params["train"].get("random_state", 42)

    # Leer datos de entrenamiento transformados
    train_df = pd.read_csv("data/processed/train_processed.csv")
    X_train = train_df.drop("target", axis=1).values
    y_train = train_df["target"].values

    # Definir modelos a evaluar
    models = {
        "logistic_regression": LogisticRegression(solver="liblinear", random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "svm": SVC(probability=True, random_state=random_state)
    }
    results = {}
    best_score = -np.inf
    best_model_name = None
    best_model = None

    # Evaluar cada modelo con validación cruzada
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="roc_auc")
        mean_score = scores.mean()
        results[name] = mean_score
        print(f"Modelo: {name}, ROC AUC (CV): {mean_score:.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_model_name = name
            best_model = model

    # Entrenar el mejor modelo en todo el conjunto de entrenamiento
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, "models/best_model.pkl")

    # Guardar métricas de entrenamiento en la carpeta evaluation
    metrics = {
        "best_model": best_model_name,
        "cv_roc_auc": best_score,
        "all_models": results
    }
    # Asegurarse de que exista la carpeta evaluation
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/metrics_train.json", "w") as f:
        json.dump(metrics, f)
    print("Entrenamiento completado. Mejor modelo:", best_model_name, "con ROC AUC:", best_score)

if __name__ == "__main__":
    main()
