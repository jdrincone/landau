#!/usr/bin/env python3
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def main():
    # Leer los datasets crudos generados
    train_df = pd.read_csv("data/raw/train.csv")
    test_df = pd.read_csv("data/raw/test.csv")
    
    # Separar features y target para el conjunto de entrenamiento
    X_train = train_df.drop("target", axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop("target", axis=1)
    
    # Aplicar StandardScaler para escalar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reconstruir DataFrames con los datos escalados
    train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    train_scaled_df["target"] = y_train.values
    
    test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    test_scaled_df["target"] = test_df["target"].values
    
    # Crear la carpeta de salida para los datos procesados si no existe
    os.makedirs("data/processed", exist_ok=True)
    train_scaled_df.to_csv("data/processed/train_processed.csv", index=False)
    test_scaled_df.to_csv("data/processed/test_processed.csv", index=False)
    
    # Guardar el escalador para uso futuro en producción o testeo
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    
    print("Datos transformados y guardados en data/processed/")

if __name__ == "__main__":
    main()
