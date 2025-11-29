"""
Probar el preprocesamiento corregido (sin invertir)
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


def preprocess_drawing_FIXED(img):
    """Preprocesamiento CORREGIDO - sin invertir"""
    img_gray = ImageOps.grayscale(img)

    # NO invertir - mantener fondo blanco y trazos negros
    img_array = np.array(img_gray)

    # Encontrar región con contenido (píxeles negros)
    inverted_for_detection = 255 - img_array
    coords = cv2.findNonZero(inverted_for_detection)
    if coords is None:
        return np.ones((28, 28), dtype=np.float32)

    x, y, w, h = cv2.boundingRect(coords)

    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img_array.shape[1] - x, w + 2 * padding)
    h = min(img_array.shape[0] - y, h + 2 * padding)

    cropped = img_array[y : y + h, x : x + w]

    max_dim = max(w, h)
    square_img = np.ones((max_dim, max_dim), dtype=np.uint8) * 255

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
    processed = preprocess_drawing_FIXED(img)

    print(f"Imagen procesada:")
    print(f"  Shape: {processed.shape}")
    print(f"  Min: {processed.min():.4f}, Max: {processed.max():.4f}")
    print(f"  Mean: {processed.mean():.4f} (dataset típico: ~0.80)")
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
        marker = "✓" if char == char_name else " "
        print(f"  {marker} {char}: {prob:.2f}%")

    print()
    return predicted_char == char_name


# Test 1: Número 5
def draw_5(draw):
    draw.line([(100, 100), (250, 100)], fill="black", width=30)
    draw.line([(100, 100), (100, 200)], fill="black", width=30)
    draw.line([(100, 200), (250, 200)], fill="black", width=30)
    draw.line([(250, 200), (250, 300)], fill="black", width=30)
    draw.line([(100, 300), (250, 300)], fill="black", width=30)


# Test 2: Letra O
def draw_O(draw):
    draw.ellipse([(100, 100), (300, 300)], outline="black", width=30)


# Test 3: Número 0
def draw_0(draw):
    draw.ellipse([(120, 100), (280, 300)], outline="black", width=30)


# Test 4: Número 1
def draw_1(draw):
    draw.line([(200, 100), (200, 300)], fill="black", width=30)


# Ejecutar tests
print("=" * 60)
print("PRUEBA CON PREPROCESAMIENTO CORREGIDO")
print("=" * 60)
print()

results = []
results.append(("5", test_character("5", draw_5)))
results.append(("O", test_character("O", draw_O)))
results.append(("0", test_character("0", draw_0)))
results.append(("1", test_character("1", draw_1)))

print("=" * 60)
print("RESUMEN DE RESULTADOS")
print("=" * 60)
for char, success in results:
    status = "✓ CORRECTO" if success else "✗ INCORRECTO"
    print(f"  {char}: {status}")

success_count = sum([s for _, s in results])
total = len(results)
print(
    f"\nPredicciones correctas: {success_count}/{total} ({success_count / total * 100:.1f}%)"
)

if success_count >= total * 0.5:
    print("\n✓ ¡Mucho mejor! El problema de 'Q' constante está resuelto.")
else:
    print("\n⚠ Aún hay problemas. Puede ser que los dibujos necesiten más detalle.")
