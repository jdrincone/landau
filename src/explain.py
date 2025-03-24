#!/usr/bin/env python3
import os
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

def main():
    # Asegurarse de que exista la carpeta evaluation
    os.makedirs("evaluation", exist_ok=True)
    
    # Cargar datos preprocesados (usamos el conjunto de entrenamiento para background)
    df = pd.read_csv("data/processed/train_processed.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    
    # Cargar el modelo calibrado
    model = joblib.load("models/best_model_calibrated.pkl")
    
    # Seleccionar un subconjunto de datos como background (por ejemplo, 50 muestras)
    background = X.sample(50, random_state=42)
    
    # Definir una función que retorne únicamente la probabilidad de la clase 1
    predict_class1 = lambda x: model.predict_proba(x)[:, 1]
    
    # Usar KernelExplainer con la función definida y el array de background
    explainer = shap.KernelExplainer(predict_class1, background.values)
    
    # Calcular los valores SHAP para el conjunto completo; shap_values tendrá shape (n_samples, n_features)
    shap_values = explainer.shap_values(X.values, nsamples=100)
    
    # 1. SHAP Summary Plot (beeswarm) para ver la distribución de valores SHAP
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    summary_path = os.path.join("evaluation", "shap_summary.png")
    plt.savefig(summary_path, bbox_inches="tight")
    plt.close()
    
    # 2. SHAP Dependence Plot para la primera característica
    plt.figure()
    shap.dependence_plot(X.columns[0], shap_values, X, show=False)
    dependence_path = os.path.join("evaluation", "shap_dependence.png")
    plt.savefig(dependence_path, bbox_inches="tight")
    plt.close()
    
    # 3. SHAP Force Plot para la primera muestra, exportado a HTML
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=False)
    force_path = os.path.join("evaluation", "shap_force.html")
    shap.save_html(force_path, force_plot)
    
    # 4. SHAP Feature Importance Bar Plot: muestra la importancia global de cada característica
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    bar_path = os.path.join("evaluation", "shap_feature_importance.png")
    plt.savefig(bar_path, bbox_inches="tight")
    plt.close()
    
    print("Gráficos SHAP generados y guardados en la carpeta 'evaluation':")
    print(" - Summary Plot:", summary_path)
    print(" - Dependence Plot:", dependence_path)
    print(" - Force Plot (HTML):", force_path)
    print(" - Feature Importance Bar Plot:", bar_path)

if __name__ == "__main__":
    main()