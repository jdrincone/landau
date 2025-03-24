#!/usr/bin/env python3
import os
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve

def main():
    # Leer datos de test transformados
    test_df = pd.read_csv("data/processed/test_processed.csv")
    X_test = test_df.drop("target", axis=1).values
    y_test = test_df["target"].values

    # Cargar el mejor modelo
    best_model = joblib.load("models/best_model.pkl")

    # Realizar predicciones
    preds = best_model.predict(X_test)
    pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    auc = roc_auc_score(y_test, pred_proba)
    acc = accuracy_score(y_test, preds)
    brier = brier_score_loss(y_test, pred_proba)

    # Calcular la matriz de confusión
    cm = confusion_matrix(y_test, preds)

    # Crear la carpeta 'evaluation' si no existe
    os.makedirs("evaluation", exist_ok=True)

    # Graficar y guardar la matriz de confusión
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = range(len(cm))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    # Añadir el número en cada celda
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max()/2. else "black")
    plt.tight_layout()
    cm_image_path = os.path.join("evaluation", "confusion_matrix.png")
    plt.savefig(cm_image_path)
    plt.close()

    # Calibración: generar la curva de calibración
    prob_true, prob_pred = calibration_curve(y_test, pred_proba, n_bins=10)
    plt.figure(figsize=(6,5))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibration curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
    plt.xlabel('Mean predicted value')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.tight_layout()
    calibration_image_path = os.path.join("evaluation", "calibration_curve.png")
    plt.savefig(calibration_image_path)
    plt.close()

    # Guardar métricas en un archivo JSON en la carpeta evaluation
    metrics = {
        "test_auc": auc,
        "test_accuracy": acc,
        "brier_score": brier,
        "confusion_matrix": cm.tolist(),  # Se guarda en formato lista para JSON
        "confusion_matrix_image": cm_image_path,
        "calibration_curve_image": calibration_image_path
    }
    metrics_path = os.path.join("evaluation", "metrics_test.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    print("Evaluación completada.")
    print(f"AUC: {auc:.4f}, Accuracy: {acc:.4f}, Brier Score: {brier:.4f}")
    print("Imagen de la matriz de confusión guardada en:", cm_image_path)
    print("Imagen de la curva de calibración guardada en:", calibration_image_path)

if __name__ == "__main__":
    main()
