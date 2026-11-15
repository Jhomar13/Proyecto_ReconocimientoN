
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. CONFIGURACION
# ==========================================

EPOCHS = 5
BATCH_SIZE = 128

os.makedirs("modelos", exist_ok=True)
os.makedirs("resultados", exist_ok=True)

# ==========================================
# 2. CARGAR DATASET MNIST
# ==========================================

# x_train: 60.000 imagenes de entrenamiento
# x_test: 10.000 imagenes de prueba
print("Cargando dataset MNIST...")

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

print("Imagenes de entrenamiento:", x_train.shape)
print("Imagenes de prueba:", x_test.shape)

# ==========================================
# 3. PREPROCESAMIENTO
# ==========================================

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Agregar canal de color
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print("Forma final:", x_train.shape)

# ==========================================
# 4. CONSTRUIR ARQUITECTURA CNN
# ==========================================

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    layers.Dropout(0.25),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dense(
        10,
        activation="softmax"
    )
])

# Mostrar arquitectura
print("\nRESUMEN DEL MODELO")
model.summary()

# ==========================================
# 5. COMPILAR MODELO
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# 6. ENTRENAR MODELO
# ==========================================

print("\nIniciando entrenamiento...")

inicio = time.perf_counter()

history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)

fin = time.perf_counter()

tiempo_entrenamiento = fin - inicio
# ==========================================
# PRECISION DE ENTRENAMIENTO Y VALIDACION
# ==========================================

accuracy_entrenamiento = history.history["accuracy"][-1]
accuracy_validacion = history.history["val_accuracy"][-1]

print("\nPRECISION DEL ENTRENAMIENTO")
print("Accuracy entrenamiento:",
      round(accuracy_entrenamiento * 100, 2), "%")

print("Accuracy validacion:",
      round(accuracy_validacion * 100, 2), "%")

print("\nTiempo de entrenamiento:",
      round(tiempo_entrenamiento, 2), "segundos")

# ==========================================
# 7. EVALUAR MODELO
# ==========================================

print("\nEvaluando modelo...")

loss, accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("Loss:", round(loss, 4))
print("Accuracy:", round(accuracy * 100, 2), "%")

# ==========================================
# 8. GENERAR PREDICCIONES
# ==========================================

y_probabilidades = model.predict(
    x_test,
    verbose=0
)

y_pred = np.argmax(y_probabilidades, axis=1)

# Reporte de clasificacion
print("\nREPORTE DE CLASIFICACION")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)

# Matriz de confusion
matriz = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(9, 7))

plt.imshow(matriz, cmap="Blues")
plt.title("Matriz de confusion - MNIST")
plt.xlabel("Prediccion")
plt.ylabel("Valor real")
plt.colorbar()

plt.savefig(
    "resultados/matriz_confusion.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 9. GRAFICA ACCURACY
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Entrenamiento"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validacion"
)

plt.title("Accuracy del modelo CNN")
plt.xlabel("Epocas")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.savefig(
    "resultados/accuracy.png",
    dpi=150
)

plt.close()

# ==========================================
# 10. GRAFICA LOSS
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Entrenamiento"
)

plt.plot(
    history.history["val_loss"],
    label="Validacion"
)

plt.title("Loss del modelo CNN")
plt.xlabel("Epocas")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.savefig(
    "resultados/loss.png",
    dpi=150
)

plt.close()

# ==========================================
# 11. GUARDAR MODELO
# ==========================================

ruta_modelo = "modelos/cnn_mnist.keras"

model.save(ruta_modelo)

tamano_mb = os.path.getsize(ruta_modelo) / (1024 * 1024)

print("\nModelo guardado:", ruta_modelo)
print("Tamano del modelo:", round(tamano_mb, 2), "MB")

print("\nPROCESO FINALIZADO CORRECTAMENTE")
