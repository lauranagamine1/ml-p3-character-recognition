# Proyecto 3: Reconocimiento de Caracteres EMNIST

## Descripción

Este proyecto implementa un sistema completo de reconocimiento de caracteres manuscritos utilizando el dataset EMNIST (Extended MNIST). El sistema incluye preprocesamiento de datos, entrenamiento de un clasificador de **Regresión Logística Multinomial**, y una interfaz gráfica interactiva para clasificar caracteres dibujados a mano.

### Características Principales

- **62 clases**: Dígitos (0-9), letras mayúsculas (A-Z) y letras minúsculas (a-z)
- **697,932 imágenes de entrenamiento** y **116,323 imágenes de test**
- **Regresión Logística Multinomial** como clasificador
- **Reducción de dimensionalidad con PCA** (784 → 61 características)
- **Interfaz gráfica interactiva** para dibujar y clasificar caracteres en tiempo real

---

## Estructura del Proyecto

```
ml-p3-character-recognition/
│
├── data/                                  # Datos EMNIST originales (CSV)
│   ├── emnist-byclass-train.csv
│   ├── emnist-byclass-test.csv
│   └── emnist-byclass-mapping.txt
│
├── preprocesamiento/                      # Preprocesamiento de datos
│   ├── preprocesamiento.ipynb            # Notebook de preprocesamiento + EDA
│   ├── class_mapping.pkl                 # Mapeo de clases (label → carácter)
│   └── output/
│       ├── emnist_train_processed.npz    # Train sin PCA (697,932 × 28 × 28)
│       ├── emnist_train_reduced.npz      # Train con PCA (697,932 × 61)
│       ├── emnist_test_processed.npz     # Test sin PCA (116,323 × 28 × 28)
│       └── emnist_test_reduced.npz       # Test con PCA (116,323 × 61)
│
├── classification/                        # Modelos de clasificación
│   └── logistic-regression/
│       ├── logistic_regression.ipynb     # Notebook de entrenamiento y evaluación
│       └── output/
│           ├── logistic_regression_model.pkl  # Modelo entrenado
│           ├── class_mapping.pkl              # Mapeo de clases
│           └── metrics.pkl                    # Métricas del modelo
│
├── graphic-interface/                     # Interfaz gráfica
│   ├── character_recognizer_gui.py       # Aplicación principal
│   └── requirements.txt                  # Dependencias
│
├── README.md                              # Este archivo
├── QUICKSTART.md                          # Guía rápida de inicio
└── check_installation.py                 # Script de verificación
```

---

## Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalar Dependencias

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter pillow opencv-python
```

O por separado:

**Para entrenamiento:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

**Para interfaz gráfica:**
```bash
pip install numpy pillow opencv-python scikit-learn
```

---

## Uso del Sistema

### 1. Preprocesamiento de Datos (Ya completado)

Los datos ya están preprocesados y disponibles en `preprocesamiento/output/`:
- **Train con PCA**: `emnist_train_reduced.npz` (697,932 × 61)
- **Test con PCA**: `emnist_test_reduced.npz` (116,323 × 61)

El preprocesamiento incluye:
- Binarización con umbral de Otsu
- Normalización
- PCA (784 dimensiones → 61 componentes)

Si deseas revisar el proceso:
```bash
cd preprocesamiento
jupyter notebook preprocesamiento.ipynb
```

### 2. Entrenamiento del Modelo de Regresión Logística

Para entrenar el modelo:

```bash
cd classification/logistic-regression
jupyter notebook logistic_regression.ipynb
```

**El notebook incluye:**

1. **Carga de Datos**
   - Datos con PCA ya aplicado (61 features)
   - 697,932 muestras de entrenamiento
   - 116,323 muestras de test

2. **Entrenamiento**
   - Regresión Logística Multinomial
   - Solver: `lbfgs` (optimizado para multiclase)
   - Regularización L2 (C=1.0)
   - Entrenamiento con paralelización

3. **Evaluación Completa**
   - Métricas globales: Accuracy, Precision, Recall, F1-Score
   - Reporte de clasificación por clase
   - Matriz de confusión y versión normalizada
   - Análisis de errores más frecuentes
   - Distribución de confianza en predicciones

4. **Guardado del Modelo**
   - Modelo entrenado
   - Mapeo de clases
   - Métricas de rendimiento

**Ejecuta todas las celdas** (Cell → Run All) para entrenar y guardar el modelo.

### 3. Interfaz Gráfica

Para ejecutar la interfaz gráfica interactiva:

```bash
cd graphic-interface
python character_recognizer_gui.py
```

**Funcionalidades:**

- **Canvas de Dibujo**: Dibuja caracteres con el mouse
- **Control de Grosor**: Ajusta el tamaño del pincel (5-40 px)
- **Botón Limpiar**: Reinicia el canvas
- **Botón Clasificar**: Procesa y clasifica el carácter dibujado
- **Predicción Principal**: Muestra el carácter con mayor confianza
- **Top 5 Predicciones**: Lista las 5 predicciones más probables
- **Imagen Procesada**: Visualiza la imagen después del preprocesamiento (28×28)
- **Información del Modelo**: Muestra métricas de rendimiento

**Cómo usar:**

1. Dibuja un dígito (0-9) o letra (A-Z, a-z) en el área blanca
2. Ajusta el grosor del pincel si es necesario
3. Haz clic en "🔍 Clasificar"
4. Observa el resultado, confianza y top 5 predicciones
5. Usa "🗑 Limpiar" para dibujar otro carácter

---

## Rendimiento del Modelo

### Métricas de Evaluación

El modelo se evalúa con múltiples métricas:
- **Accuracy**: Proporción de predicciones correctas
- **Precision**: Exactitud de las predicciones positivas
- **Recall**: Cobertura de las clases verdaderas
- **F1-Score**: Media armónica de Precision y Recall

*(Los valores exactos se generan al ejecutar el notebook y se muestran en la interfaz)*

### Características del Modelo

- **Algoritmo**: Regresión Logística Multinomial
- **Solver**: lbfgs (optimizado para multiclase)
- **Regularización**: L2 (C=1.0)
- **Clases**: 62 (0-9, A-Z, a-z)
- **Features**: 61 (reducidas con PCA)
- **Varianza explicada por PCA**: ~95%

### Análisis de Resultados

**Fortalezas:**
- ✅ Entrenamiento rápido (~5-15 minutos)
- ✅ Predicción en tiempo real
- ✅ Buen rendimiento general en dígitos
- ✅ Modelo interpretable (coeficientes lineales)
- ✅ Bajo costo computacional

**Desafíos comunes:**
- Confusión entre caracteres visualmente similares:
  - O (letra) vs 0 (cero)
  - I (mayúscula) vs l (L minúscula) vs 1 (uno)
  - S vs 5
  - Z vs 2
  - Mayúsculas vs minúsculas similares (C/c, O/o, S/s)

---

## Justificación de Decisiones de Diseño

### 1. Preprocesamiento

**Binarización con Umbral de Otsu:**
- Reduce ruido y variabilidad en las imágenes
- Mejora la robustez del modelo
- Estandariza las imágenes a valores binarios (0 o 1)

**PCA (784 → 61 dimensiones):**
- **Eficiencia**: Reduce tiempo de entrenamiento ~92%
- **Previene overfitting**: Menos parámetros a aprender
- **Elimina ruido**: Mantiene solo componentes con mayor varianza
- **Preserva información**: ~95% de varianza explicada
- **Compatibilidad**: Mejora convergencia de algoritmos lineales

### 2. Regresión Logística Multinomial

**¿Por qué Regresión Logística?**

✅ **Ventajas:**
- **Eficiente**: Rápido para datasets grandes (697K muestras)
- **Multiclase nativo**: Maneja 62 clases con `multi_class='multinomial'`
- **Predicción en tiempo real**: Ideal para interfaz interactiva
- **Interpretable**: Coeficientes lineales por clase
- **Regularización L2**: Previene overfitting automáticamente
- **Funciona bien con PCA**: Asume linealidad en espacio transformado
- **Convergencia estable**: Solver lbfgs es robusto

❌ **Limitaciones:**
- Asume relaciones lineales (fronteras de decisión lineales)
- Rendimiento inferior a modelos más complejos (CNNs, XGBoost)
- Dificultad con caracteres muy similares

**Parámetros seleccionados:**
- `solver='lbfgs'`: Optimizado para problemas multiclase grandes
- `multi_class='multinomial'`: Modelo multinomial (softmax)
- `C=1.0`: Regularización L2 estándar
- `max_iter=1000`: Suficientes iteraciones para convergencia
- `n_jobs=-1`: Paralelización usando todos los cores

### 3. Interfaz Gráfica

**Tecnología: Tkinter**
- Biblioteca estándar de Python (sin instalación adicional)
- Multiplataforma (Windows, Mac, Linux)
- Suficiente para aplicación interactiva

**Preprocesamiento en tiempo real:**
- Detección automática de región de interés (OpenCV)
- Centrado del carácter
- Redimensionamiento a 28×28
- Binarización con umbral

**Características de usabilidad:**
- Control de grosor: Adaptabilidad a estilos de dibujo
- Visualización procesada: Transparencia del preprocesamiento
- Top 5 predicciones: Contexto sobre alternativas