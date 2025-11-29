"""
Script para debuggear el preprocesamiento de imágenes
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Cargar componentes
prepro_dir = Path("../preprocesamiento")
model_dir = Path("../classification/logistic-regression/output")

print("Cargando componentes...")
with open(prepro_dir / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(prepro_dir / "pca.pkl", "rb") as f:
    pca = pickle.load(f)

with open(model_dir / "logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(model_dir / "class_mapping.pkl", "rb") as f:
    class_mapping = pickle.load(f)

print("✓ Componentes cargados")

# Cargar una imagen de prueba real del dataset
print("\nCargando imagen de prueba del dataset...")
test_data = np.load("../preprocesamiento/output/emnist_test_processed.npz")
X_test = test_data["X_test"]
y_test = test_data["y_test"]

# Tomar una imagen (por ejemplo, el primer '5')
idx_5 = np.where(y_test == 5)[0][0]
test_image = X_test[idx_5]  # Shape: (28, 28)
print(f"Imagen de prueba: clase {y_test[idx_5]} ({class_mapping[y_test[idx_5]]})")

# Simular el procesamiento de la GUI
print("\n" + "=" * 60)
print("SIMULANDO PIPELINE DE LA GUI")
print("=" * 60)

# Paso 1: Aplanar
img_flat = test_image.reshape(1, -1).astype(np.float64)
print(f"\n1. Imagen aplanada: {img_flat.shape}")
print(
    f"   Min: {img_flat.min():.4f}, Max: {img_flat.max():.4f}, Mean: {img_flat.mean():.4f}"
)

# Paso 2: Escalar
img_scaled = scaler.transform(img_flat)
print(f"\n2. Imagen escalada: {img_scaled.shape}")
print(
    f"   Min: {img_scaled.min():.4f}, Max: {img_scaled.max():.4f}, Mean: {img_scaled.mean():.4f}"
)

# Paso 3: PCA
img_pca = pca.transform(img_scaled)
print(f"\n3. Imagen con PCA: {img_pca.shape}")
print(
    f"   Min: {img_pca.min():.4f}, Max: {img_pca.max():.4f}, Mean: {img_pca.mean():.4f}"
)

# Paso 4: Predecir
prediction = model.predict(img_pca)[0]
probabilities = model.predict_proba(img_pca)[0]

print(f"\n4. Predicción:")
print(f"   Clase predicha: {prediction} ({class_mapping[prediction]})")
print(f"   Confianza: {probabilities[prediction] * 100:.2f}%")

# Top 5
top5_indices = np.argsort(probabilities)[-5:][::-1]
print(f"\n   Top 5:")
for idx in top5_indices:
    print(f"      {class_mapping[idx]}: {probabilities[idx] * 100:.2f}%")

print("\n" + "=" * 60)
print("PROBANDO CON UNA IMAGEN VACÍA (CASO PROBLEMÁTICO)")
print("=" * 60)

# Crear una imagen casi vacía (lo que podría pasar si el dibujo no se procesa bien)
empty_image = np.zeros((28, 28), dtype=np.float32)
print(f"\nImagen vacía: {empty_image.shape}")

img_flat_empty = empty_image.reshape(1, -1).astype(np.float64)
img_scaled_empty = scaler.transform(img_flat_empty)
img_pca_empty = pca.transform(img_scaled_empty)

prediction_empty = model.predict(img_pca_empty)[0]
probabilities_empty = model.predict_proba(img_pca_empty)[0]

print(f"\nPredicción con imagen vacía:")
print(f"   Clase predicha: {prediction_empty} ({class_mapping[prediction_empty]})")
print(f"   Confianza: {probabilities_empty[prediction_empty] * 100:.2f}%")

# Top 5
top5_empty = np.argsort(probabilities_empty)[-5:][::-1]
print(f"\n   Top 5:")
for idx in top5_empty:
    print(f"      {class_mapping[idx]}: {probabilities_empty[idx] * 100:.2f}%")

# Ver qué valores tiene la imagen después del preprocesamiento
print("\n" + "=" * 60)
print("ANÁLISIS DE TRANSFORMACIÓN")
print("=" * 60)
print(f"\nImagen original (dataset):")
print(f"  Range: [{test_image.min():.4f}, {test_image.max():.4f}]")
print(f"  Mean: {test_image.mean():.4f}")
print(f"  Std: {test_image.std():.4f}")
print(f"  Non-zero pixels: {(test_image > 0).sum()} / {test_image.size}")

print(f"\nImagen vacía:")
print(f"  Range: [{empty_image.min():.4f}, {empty_image.max():.4f}]")
print(f"  Mean: {empty_image.mean():.4f}")
print(f"  Std: {empty_image.std():.4f}")
print(f"  Non-zero pixels: {(empty_image > 0).sum()} / {empty_image.size}")

print("\n✓ Prueba completada")
