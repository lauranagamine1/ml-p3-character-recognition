"""
Test completo del pipeline de la GUI con el preprocesamiento corregido
"""

import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

# Cargar componentes
prepro_dir = Path("../preprocesamiento")
model_dir = Path("../classification/logistic-regression/output")

print("Cargando modelo y transformadores...")
with open(prepro_dir / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(prepro_dir / "pca.pkl", "rb") as f:
    pca = pickle.load(f)

with open(model_dir / "logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(model_dir / "class_mapping.pkl", "rb") as f:
    class_mapping = pickle.load(f)

print("✓ Componentes cargados\n")


def preprocess_drawing(img):
    """Mismo preprocesamiento que usa la GUI"""
    img_gray = ImageOps.grayscale(img)
    img_inverted = ImageOps.invert(img_gray)
    img_array = np.array(img_inverted)

    coords = cv2.findNonZero(img_array)
    if coords is None:
        return np.zeros((28, 28), dtype=np.float32)

    x, y, w, h = cv2.boundingRect(coords)

    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img_array.shape[1] - x, w + 2 * padding)
    h = min(img_array.shape[0] - y, h + 2 * padding)

    cropped = img_array[y : y + h, x : x + w]

    max_dim = max(w, h)
    square_img = np.zeros((max_dim, max_dim), dtype=np.uint8)

    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    square_img[y_offset : y_offset + h, x_offset : x_offset + w] = cropped

    resized = cv2.resize(square_img, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0

    return normalized


def test_character(char_name, draw_function):
    """Prueba la predicción de un carácter"""
    print(f"=" * 60)
    print(f"Probando: {char_name}")
    print(f"=" * 60)

    # Crear imagen y dibujar
    img = Image.new("RGB", (400, 400), "white")
    draw = ImageDraw.Draw(img)
    draw_function(draw)

    # Preprocesar
    processed = preprocess_drawing(img)

    print(f"Imagen procesada:")
    print(f"  Shape: {processed.shape}")
    print(f"  Min: {processed.min():.4f}, Max: {processed.max():.4f}")
    print(f"  Mean: {processed.mean():.4f}")
    print(f"  Non-zero: {(processed > 0).sum()} / {processed.size}")

    # Predecir
    img_flat = processed.reshape(1, -1).astype(np.float64)
    img_scaled = scaler.transform(img_flat)
    img_pca = pca.transform(img_scaled)

    prediction = model.predict(img_pca)[0]
    probabilities = model.predict_proba(img_pca)[0]

    predicted_char = class_mapping[prediction]
    confidence = probabilities[prediction] * 100

    print(f"\nPredicción: {predicted_char} (confianza: {confidence:.2f}%)")

    # Top 5
    top5_indices = np.argsort(probabilities)[-5:][::-1]
    print(f"Top 5:")
    for idx in top5_indices:
        char = class_mapping[idx]
        prob = probabilities[idx] * 100
        print(f"  {char}: {prob:.2f}%")

    print()
    return predicted_char == char_name


# Test 1: Número 5
def draw_5(draw):
    draw.line([(100, 100), (250, 100)], fill="black", width=25)
    draw.line([(100, 100), (100, 200)], fill="black", width=25)
    draw.line([(100, 200), (250, 200)], fill="black", width=25)
    draw.line([(250, 200), (250, 300)], fill="black", width=25)
    draw.line([(100, 300), (250, 300)], fill="black", width=25)


# Test 2: Letra O
def draw_O(draw):
    draw.ellipse([(100, 100), (300, 300)], outline="black", width=25)


# Test 3: Letra I
def draw_I(draw):
    draw.line([(200, 100), (200, 300)], fill="black", width=25)


# Test 4: Número 0
def draw_0(draw):
    draw.ellipse([(120, 100), (280, 300)], outline="black", width=25)


# Ejecutar tests
results = []
results.append(test_character("5", draw_5))
results.append(test_character("O", draw_O))
results.append(test_character("I", draw_I))
results.append(test_character("0", draw_0))

print("=" * 60)
print("RESUMEN DE RESULTADOS")
print("=" * 60)
success = sum(results)
total = len(results)
print(f"Predicciones correctas: {success}/{total}")
print(f"Tasa de éxito: {success / total * 100:.1f}%")

if success == total:
    print("\n✓ ¡Todos los tests pasaron! La GUI debería funcionar correctamente ahora.")
else:
    print("\n⚠ Algunos tests fallaron, pero esto es normal con dibujos simples.")
    print("  La GUI funcionará mejor con dibujos más detallados del usuario.")
