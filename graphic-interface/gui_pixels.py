"""
Interfaz Gráfica Simplificada para Reconocimiento de Caracteres
Dibuja directamente en 28x28 píxeles (con zoom visual)
"""

import pickle
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
from skimage.filters import threshold_otsu


class SimpleCharacterRecognizerGUI:
    """
    Interfaz simple que dibuja directamente en resolución 28x28.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Reconocedor de Caracteres - 28x28 Direct")
        self.root.geometry("800x700")
        self.root.resizable(False, False)

        # Canvas real de 28x28, mostrado con zoom
        self.pixel_size = 15  # Cada píxel se muestra como 15x15
        self.canvas_pixels = 28
        self.display_size = self.pixel_size * self.canvas_pixels  # 420x420

        # Matriz de píxeles 28x28 (fondo blanco = 255, trazo negro = 0)
        self.pixels = np.ones((28, 28), dtype=np.uint8) * 255

        # Para dibujo continuo
        self.is_drawing = False
        self.brush_size = 2  # Radio en píxeles

        # Cargar modelo
        self.load_model()

        # Configurar interfaz
        self.setup_ui()

    def load_model(self):
        """Carga el modelo y componentes."""
        model_dir = Path("../classification/xgBoost/output")
        prepro_dir = Path("../preprocesamiento")

        try:
            with open(model_dir / "xgboost_model.pkl", "rb") as f:
                self.model = pickle.load(f)
            with open(model_dir / "class_mapping.pkl", "rb") as f:
                self.class_mapping = pickle.load(f)
            with open(model_dir / "xgboost_metrics.pkl", "rb") as f:
                self.metrics = pickle.load(f)
            with open(prepro_dir / "scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            with open(prepro_dir / "pca.pkl", "rb") as f:
                self.pca = pickle.load(f)
            print("✓ Modelo cargado correctamente")
        except FileNotFoundError as e:
            messagebox.showerror("Error", f"No se encontraron archivos del modelo: {e}")
            raise e

    def setup_ui(self):
        """Configura la interfaz."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Título
        title_label = ttk.Label(
            main_frame,
            text="Reconocedor de Caracteres - Canvas 28x28",
            font=("Arial", 18, "bold"),
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # Frame izquierdo: Canvas
        left_frame = ttk.LabelFrame(
            main_frame, text="Dibuja Aquí (28x28 píxeles)", padding="10"
        )
        left_frame.grid(row=1, column=0, padx=10, pady=10, sticky=(tk.N, tk.S))

        # Canvas con zoom visual
        self.canvas = tk.Canvas(
            left_frame,
            width=self.display_size,
            height=self.display_size,
            bg="white",
            cursor="crosshair",
        )
        self.canvas.pack()

        # Dibujar grid
        self.draw_grid()

        # Eventos del mouse
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_pixel)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        # Controles
        controls_frame = ttk.Frame(left_frame)
        controls_frame.pack(pady=10)

        ttk.Label(controls_frame, text="Tamaño del pincel:").grid(
            row=0, column=0, padx=5
        )
        self.brush_label = ttk.Label(controls_frame, text=f"{self.brush_size} px")
        self.brush_label.grid(row=0, column=2, padx=5)

        self.brush_slider = ttk.Scale(
            controls_frame,
            from_=1,
            to=5,
            orient=tk.HORIZONTAL,
            length=150,
            command=self.update_brush_size,
        )
        self.brush_slider.set(self.brush_size)
        self.brush_slider.grid(row=0, column=1, padx=5)

        # Botones
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
        right_frame.grid(row=1, column=1, padx=10, pady=10, sticky=(tk.N, tk.S))

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

        # Top 5
        top5_frame = ttk.LabelFrame(
            right_frame, text="Top 5 Predicciones", padding="10"
        )
        top5_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        columns = ("Carácter", "Confianza")
        self.top5_tree = ttk.Treeview(
            top5_frame, columns=columns, show="headings", height=5
        )
        self.top5_tree.heading("Carácter", text="Carácter")
        self.top5_tree.heading("Confianza", text="Confianza")
        self.top5_tree.column("Carácter", width=100, anchor=tk.CENTER)
        self.top5_tree.column("Confianza", width=100, anchor=tk.CENTER)
        self.top5_tree.pack()

        # Info del modelo
        info_frame = ttk.LabelFrame(
            main_frame, text="Información del Modelo", padding="10"
        )
        info_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        info_text = (
            f"Modelo: XGBoost  |  "
            f"Accuracy: {self.metrics['test_accuracy'] * 100:.2f}%  |  "
            f"Resolución: 28x28 píxeles directos"
        )
        ttk.Label(info_frame, text=info_text, font=("Arial", 10)).pack()

    def draw_grid(self):
        """Dibuja una cuadrícula sutil para ver los píxeles."""
        for i in range(self.canvas_pixels + 1):
            x = i * self.pixel_size
            self.canvas.create_line(
                x, 0, x, self.display_size, fill="lightgray", width=1
            )
            self.canvas.create_line(
                0, x, self.display_size, x, fill="lightgray", width=1
            )

    def update_brush_size(self, value):
        """Actualiza el tamaño del pincel."""
        self.brush_size = int(float(value))
        self.brush_label.config(text=f"{self.brush_size} px")

    def start_draw(self, event):
        """Inicia el dibujo."""
        self.is_drawing = True
        self.draw_pixel(event)

    def draw_pixel(self, event):
        """Dibuja píxeles en la matriz."""
        if not self.is_drawing:
            return

        # Convertir coordenadas del canvas a coordenadas de píxel
        px = event.x // self.pixel_size
        py = event.y // self.pixel_size

        # Validar límites
        if 0 <= px < self.canvas_pixels and 0 <= py < self.canvas_pixels:
            # Dibujar con el tamaño de pincel
            for dx in range(-self.brush_size + 1, self.brush_size):
                for dy in range(-self.brush_size + 1, self.brush_size):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < self.canvas_pixels and 0 <= ny < self.canvas_pixels:
                        # Pintar de negro (0)
                        self.pixels[ny, nx] = 0
                        self.update_pixel_display(nx, ny)

    def update_pixel_display(self, px, py):
        """Actualiza la visualización de un píxel en el canvas."""
        x1 = px * self.pixel_size
        y1 = py * self.pixel_size
        x2 = x1 + self.pixel_size
        y2 = y1 + self.pixel_size

        color = "black" if self.pixels[py, px] == 0 else "white"
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    def stop_draw(self, event):
        """Detiene el dibujo."""
        self.is_drawing = False

    def clear_canvas(self):
        """Limpia el canvas."""
        self.pixels = np.ones((28, 28), dtype=np.uint8) * 255
        self.canvas.delete("all")
        self.draw_grid()
        self.prediction_label.config(text="?")
        self.confidence_label.config(text="---%")
        self.top5_tree.delete(*self.top5_tree.get_children())

    def preprocess_image(self):
        """
        Preprocesa la imagen de 28x28.
        Aplica el mismo preprocesamiento que el dataset.
        """
        # La imagen ya está en 28x28
        # Invertir polaridad (fondo negro, letra blanca)
        inverted = 255 - self.pixels

        # Normalizar a [0, 1]
        normalized = inverted.astype(np.float32) / 255.0

        # Binarización con Otsu
        thresh = threshold_otsu(normalized)
        binarized = (normalized >= thresh).astype(np.uint8)

        return binarized

    def classify(self):
        """Clasifica el carácter dibujado."""
        # Preprocesar
        processed_img = self.preprocess_image()

        # Verificar que hay algo dibujado
        if processed_img.sum() == 0:
            messagebox.showwarning(
                "Advertencia", "Por favor, dibuja un carácter antes de clasificar."
            )
            return

        # Aplanar (28x28 -> 784)
        img_flat = processed_img.reshape(1, -1).astype(np.float64)

        # Aplicar scaler y PCA
        img_scaled = self.scaler.transform(img_flat)
        img_pca = self.pca.transform(img_scaled)

        # Predecir
        prediction = self.model.predict(img_pca)[0]
        probabilities = self.model.predict_proba(img_pca)[0]

        # Obtener top 5
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


def main():
    """Función principal."""
    root = tk.Tk()
    app = SimpleCharacterRecognizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
