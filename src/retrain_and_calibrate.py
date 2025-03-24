#!/usr/bin/env python3
import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

def main():
    # Leer los datos preprocesados
    data = pd.read_csv("data/processed/train_processed.csv")
    X = data.drop("target", axis=1).values
    y = data["target"].values

    # Entrenar el modelo base
    base_model = LogisticRegression(solver="liblinear", random_state=42)
    base_model.fit(X, y)

    # Calibrar el modelo
    # Nota: Se pasa el modelo como primer argumento en lugar de usar el parámetro 'base_estimator'
    calibrated_model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
    calibrated_model.fit(X, y)

    # Crear carpeta de salida si no existe
    os.makedirs("models", exist_ok=True)
    # Guardar el modelo calibrado
    joblib.dump(calibrated_model, "models/best_model_calibrated.pkl")
    print("Modelo reentrenado y calibrado guardado en models/best_model_calibrated.pkl")

if __name__ == "__main__":
    main()
