import numpy as np
import pandas as pd
import cv2
from skimage.filters import threshold_otsu
from PIL import Image, ImageOps


class ImageProcessor:
    """Procesador de imagen simple.

    Métodos
    -------
    process_img(img, binarize=False)
        Aplica: Gaussian blur -> ROI extraction -> centrar -> escalar a 28x28.
        Devuelve una imagen `np.ndarray` dtype=uint8 con rango 0-255 (28x28).
        Si `binarize=True` devuelve array 0/1 (uint8).
    """

    def __init__(self):
        pass

    @staticmethod
    def process_img(img) -> np.ndarray:
        # Convertir a escala de grises
        img_gray = ImageOps.grayscale(img)
        img_array = np.array(img_gray).astype(np.uint8, copy=False)

        # Aplicar Gaussian Blur para reducir ruido
        blurred = cv2.GaussianBlur(img_array, (5, 5), 0)

        # Encontrar región con contenido (píxeles negros/oscuros < 255)
        # Invertir temporalmente para encontrar el contenido dibujado
        inverted_for_detection = 255 - blurred
        coords = cv2.findNonZero(inverted_for_detection)
        if coords is None:
            # Si no hay dibujo, retornar imagen vacía (todos ceros)
            return np.zeros((28, 28), dtype=np.uint8)

        x, y, w, h = cv2.boundingRect(coords)

        # Agregar padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(blurred.shape[1] - x, w + 2 * padding)
        h = min(blurred.shape[0] - y, h + 2 * padding)

        # Recortar región de interés
        cropped = blurred[y : y + h, x : x + w]

        # Hacer la imagen cuadrada con fondo blanco
        max_dim = max(w, h)
        square_img = np.ones((max_dim, max_dim), dtype=np.uint8) * 255

        # Centrar la imagen
        y_offset = (max_dim - h) // 2
        x_offset = (max_dim - w) // 2
        square_img[y_offset : y_offset + h, x_offset : x_offset + w] = cropped

        # Redimensionar a 28x28
        resized = cv2.resize(square_img, (28, 28), interpolation=cv2.INTER_AREA)

        # Normalizar a [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Binarización con umbral de Otsu
        thresh = threshold_otsu(normalized)
        binarized = (normalized >= thresh).astype(np.uint8)

        return binarized
        