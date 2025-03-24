#!/usr/bin/env python3
import os
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def main():
    # Generar dataset sintético
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=5,
                               n_redundant=2, n_classes=2, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

    # Dividir en entrenamiento y test
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["target"])

    # Crear la carpeta si no existe y guardar los CSVs
    os.makedirs("data/raw", exist_ok=True)
    train_df.to_csv("data/raw/train.csv", index=False)
    test_df.to_csv("data/raw/test.csv", index=False)
    print("Datos generados y guardados en data/raw/")

if __name__ == "__main__":
    main()
