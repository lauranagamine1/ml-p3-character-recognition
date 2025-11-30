"""
Interfaz Gráfica para Reconocimiento de Caracteres EMNIST
usando XGBoost

Este módulo proporciona una interfaz gráfica interactiva donde el usuario
puede dibujar caracteres y el modelo de XGBoost los clasificará.
"""

import pickle
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from skimage.filters import threshold_otsu
from FeatureExtractor import FeatureExtractor


class CharacterRecognizerGUI:
    """
    Interfaz gráfica para reconocimiento de caracteres manuscritos.

    Permite al usuario dibujar en un canvas y clasificar el carácter
    dibujado usando el modelo de XGBoost entrenado.
    """

    def __init__(self, root):
        """
        Inicializa la interfaz gráfica.

        Args:
            root: Ventana raíz de Tkinter
        """
        self.root = root
        self.root.title("Reconocedor de Caracteres EMNIST")
        self.root.geometry("900x700")
        self.root.resizable(False, False)

        # Configuración del canvas de dibujo
        self.canvas_size = 400
        self.brush_size = 25

        # Variable para tracking del mouse
        self.last_x = None
        self.last_y = None

        # Cargar modelo y componentes (ANTES de setup_ui)
        self.load_model()

        # Inicializar imagen para dibujar
        self.image = Image.new("RGB", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)

        # Configurar interfaz (DESPUÉS de cargar modelo)
        self.setup_ui()

    def load_model(self):
        """Carga el modelo entrenado y componentes de preprocesamiento."""
        model_dir = Path("../classification/xgBoost/output")
        prepro_dir = Path("../preprocesamiento")

        try:
            # Cargar modelo XGBoost
            with open(model_dir / "xgboost_model.pkl", "rb") as f:
                self.model = pickle.load(f)

            # Cargar mapeo de clases
            with open(model_dir / "class_mapping.pkl", "rb") as f:
                self.class_mapping = pickle.load(f)

            # Cargar métricas
            with open(model_dir / "xgboost_metrics.pkl", "rb") as f:
                self.metrics = pickle.load(f)

            # Cargar pipeline de reducción de dimensionalidad
            with open(prepro_dir / "dim_reduction_pipeline.pkl", "rb") as f:
                self.dim_reduction = pickle.load(f)

            print("✓ Modelo y componentes cargados exitosamente")

        except FileNotFoundError as e:
            messagebox.showerror(
                "Error",
                "No se encontraron los archivos del modelo.\n"
                "Por favor, ejecuta primero el notebook de entrenamiento.",
            )
            raise e

    def setup_ui(self):
        """Configura todos los elementos de la interfaz de usuario."""
        # Estilo
        style = ttk.Style()
        style.theme_use("clam")

        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Título
        title_label = ttk.Label(
            main_frame,
            text="Reconocedor de Caracteres EMNIST",
            font=("Arial", 20, "bold"),
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # Subtítulo
        subtitle_label = ttk.Label(
            main_frame,
            text="Dibuja un dígito (0-9) o letra (A-Z, a-z)",
            font=("Arial", 12),
        )
        subtitle_label.grid(row=1, column=0, columnspan=2, pady=5)

        # Frame izquierdo: Canvas de dibujo
        left_frame = ttk.LabelFrame(main_frame, text="Área de Dibujo", padding="10")
        left_frame.grid(row=2, column=0, padx=10, pady=10, sticky=(tk.N, tk.S))

        # Canvas
        self.canvas = tk.Canvas(
            left_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="white",
            cursor="crosshair",
        )
        self.canvas.pack()

        # Eventos del mouse
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        # Controles del canvas
        canvas_controls = ttk.Frame(left_frame)
        canvas_controls.pack(pady=10)

        # Control de grosor del pincel
        ttk.Label(canvas_controls, text="Grosor del pincel:").grid(
            row=0, column=0, padx=5
        )

        # Crear brush_label ANTES del slider
        self.brush_label = ttk.Label(canvas_controls, text=f"{self.brush_size}px")
        self.brush_label.grid(row=0, column=2, padx=5)

        # Ahora crear el slider
        self.brush_slider = ttk.Scale(
            canvas_controls,
            from_=5,
            to=40,
            orient=tk.HORIZONTAL,
            length=200,
            command=self.update_brush_size,
        )
        self.brush_slider.set(self.brush_size)
        self.brush_slider.grid(row=0, column=1, padx=5)

        # Botones de control
        buttons_frame = ttk.Frame(left_frame)
        buttons_frame.pack(pady=10)

        ttk.Button(
            buttons_frame, text="🗑 Limpiar", command=self.clear_canvas, width=15
        ).grid(row=0, column=0, padx=5)

        ttk.Button(
            buttons_frame, text="🔍 Clasificar", command=self.classify, width=15
        ).grid(row=0, column=1, padx=5)

        # Frame derecho: Resultados
        right_frame = ttk.LabelFrame(main_frame, text="Resultados", padding="10")
        right_frame.grid(row=2, column=1, padx=10, pady=10, sticky=(tk.N, tk.S))

        # Resultado principal
        result_frame = ttk.Frame(right_frame)
        result_frame.pack(pady=10)

        ttk.Label(result_frame, text="Predicción:", font=("Arial", 12)).pack()
        self.prediction_label = ttk.Label(
            result_frame, text="?", font=("Arial", 60, "bold"), foreground="blue"
        )
        self.prediction_label.pack(pady=10)

        ttk.Label(result_frame, text="Confianza:", font=("Arial", 12)).pack()
        self.confidence_label = ttk.Label(result_frame, text="---%", font=("Arial", 16))
        self.confidence_label.pack()

        # Top 5 predicciones
        top5_frame = ttk.LabelFrame(
            right_frame, text="Top 5 Predicciones", padding="10"
        )
        top5_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # Tabla de top 5
        columns = ("Carácter", "Confianza")
        self.top5_tree = ttk.Treeview(
            top5_frame, columns=columns, show="headings", height=5
        )

        self.top5_tree.heading("Carácter", text="Carácter")
        self.top5_tree.heading("Confianza", text="Confianza")
        self.top5_tree.column("Carácter", width=100, anchor=tk.CENTER)
        self.top5_tree.column("Confianza", width=100, anchor=tk.CENTER)
        self.top5_tree.pack()

        # Imagen procesada
        processed_frame = ttk.LabelFrame(
            right_frame, text="Imagen Procesada (28x28)", padding="10"
        )
        processed_frame.pack(pady=10)

        self.processed_canvas = tk.Canvas(
            processed_frame, width=140, height=140, bg="white"
        )
        self.processed_canvas.pack()

        # Información del modelo
        info_frame = ttk.LabelFrame(
            main_frame, text="Información del Modelo", padding="10"
        )
        info_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        info_text = (
            f"Modelo: XGBoost  |  "
            f"Accuracy: {self.metrics['test_accuracy'] * 100:.2f}%  |  "
            f"F1-Score: {self.metrics['test_f1']:.4f}  |  "
            f"Clases: 62 (0-9, A-Z, a-z)"
        )
        ttk.Label(info_frame, text=info_text, font=("Arial", 10)).pack()

    def update_brush_size(self, value):
        """Actualiza el tamaño del pincel."""
        self.brush_size = int(float(value))
        self.brush_label.config(text=f"{self.brush_size}px")

    def start_draw(self, event):
        """Inicia el dibujo al presionar el mouse."""
        self.last_x = event.x
        self.last_y = event.y

    def draw_on_canvas(self, event):
        """Dibuja en el canvas mientras se arrastra el mouse."""
        if self.last_x and self.last_y:
            # Dibujar en el canvas de Tkinter
            self.canvas.create_line(
                self.last_x,
                self.last_y,
                event.x,
                event.y,
                width=self.brush_size,
                fill="black",
                capstyle=tk.ROUND,
                smooth=True,
            )

            # Dibujar en la imagen PIL
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill="black",
                width=self.brush_size,
            )

        self.last_x = event.x
        self.last_y = event.y

    def stop_draw(self, event):
        """Detiene el dibujo al soltar el mouse."""
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        """Limpia el canvas y reinicia los resultados."""
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)

        # Limpiar resultados
        self.prediction_label.config(text="?")
        self.confidence_label.config(text="---%")
        self.top5_tree.delete(*self.top5_tree.get_children())
        self.processed_canvas.delete("all")

    def preprocess_image(self):
        """
        Preprocesa la imagen dibujada para el modelo.

        Aplica el MISMO preprocesamiento que el dataset de entrenamiento:
        1. Conversión a escala de grises
        2. Inversión de polaridad (fondo negro, letra blanca)
        3. Normalización a [0, 1]
        4. Binarización con umbral de Otsu

        Returns:
            numpy.ndarray: Imagen procesada de 28x28 (valores 0 o 1)
        """
        from skimage.filters import threshold_otsu

        # Convertir a escala de grises
        img_gray = ImageOps.grayscale(self.image)
        img_array = np.array(img_gray)

        # Encontrar región con contenido (píxeles negros/oscuros < 255)
        # Invertir temporalmente para encontrar el contenido dibujado
        inverted_for_detection = 255 - img_array
        coords = cv2.findNonZero(inverted_for_detection)
        if coords is None:
            # Si no hay dibujo, retornar imagen vacía (todos ceros)
            return np.zeros((28, 28), dtype=np.uint8)

        x, y, w, h = cv2.boundingRect(coords)

        # Agregar padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_array.shape[1] - x, w + 2 * padding)
        h = min(img_array.shape[0] - y, h + 2 * padding)

        # Recortar región de interés
        cropped = img_array[y : y + h, x : x + w]

        # Hacer la imagen cuadrada con fondo blanco
        max_dim = max(w, h)
        square_img = np.ones((max_dim, max_dim), dtype=np.uint8) * 255

        # Centrar la imagen
        y_offset = (max_dim - h) // 2
        x_offset = (max_dim - w) // 2
        square_img[y_offset : y_offset + h, x_offset : x_offset + w] = cropped

        # Redimensionar a 28x28
        resized = cv2.resize(square_img, (28, 28), interpolation=cv2.INTER_AREA)

        # Normalizar a [0, 1] sobre la imagen invertida
        normalized = resized.astype(np.float32) / 255.0

        # Binarización con umbral de Otsu
        thresh = threshold_otsu(normalized)
        binarized = (normalized >= thresh).astype(np.uint8)

        return binarized

    def classify(self):
        """Clasifica el carácter dibujado."""
        # Preprocesar imagen
        processed_img = self.preprocess_image()

        # Verificar que hay algo dibujado
        if processed_img.sum() == 0:
            messagebox.showwarning(
                "Advertencia", "Por favor, dibuja un carácter antes de clasificar."
            )
            return

        # Mostrar imagen procesada
        self.show_processed_image(processed_img)

        # Extraer features
        img_features = FeatureExtractor().extract_features_img(processed_img)

        # El pipeline de reducción espera forma (n_samples, n_features)
        img_features = img_features.reshape(1, -1)

        # Reducir dimensionalidad
        img_reduced = self.dim_reduction.transform(img_features)

        # Predecir (ahora con 61 features)
        prediction = self.model.predict(img_reduced)[0]
        probabilities = self.model.predict_proba(img_reduced)[0]

        # Obtener top 5 predicciones
        top5_indices = np.argsort(probabilities)[-5:][::-1]

        # Actualizar interfaz
        predicted_char = self.class_mapping[prediction]
        confidence = probabilities[prediction] * 100

        self.prediction_label.config(text=predicted_char)
        self.confidence_label.config(text=f"{confidence:.2f}%")

        # Actualizar top 5
        self.top5_tree.delete(*self.top5_tree.get_children())
        for idx in top5_indices:
            char = self.class_mapping[idx]
            prob = probabilities[idx] * 100
            self.top5_tree.insert("", tk.END, values=(char, f"{prob:.2f}%"))

    def show_processed_image(self, img):
        """
        Muestra la imagen procesada en el canvas.

        Args:
            img: Imagen procesada de 28x28 (valores float en [0, 1])
        """
        self.processed_canvas.delete("all")

        # Escalar a 140x140 para visualización
        scale = 5
        for i in range(28):
            for j in range(28):
                # Convertir valor float [0, 1] a escala de grises
                intensity = int(img[i, j] * 255)
                gray_color = f"#{intensity:02x}{intensity:02x}{intensity:02x}"

                x1 = j * scale
                y1 = i * scale
                x2 = x1 + scale
                y2 = y1 + scale
                self.processed_canvas.create_rectangle(
                    x1, y1, x2, y2, fill=gray_color, outline=""
                )


def main():
    """Función principal para ejecutar la aplicación."""
    root = tk.Tk()
    app = CharacterRecognizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
