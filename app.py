import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard

# Definir parámetros
TAMANO_IMG = 100  # Tamaño de las imágenes
BATCH_SIZE = 32   # Tamaño del lote para entrenamiento
EPOCHS = 30       # Número de épocas de entrenamiento

# Directorio del dataset
DIRECTORIO_DATASET = './dataset-original'

# Configurar generadores de datos
datagen = ImageDataGenerator(
    rescale=1./255,        # Normalizar imágenes
    validation_split=0.15   # Separar 15% para validación
)

try:
    assert os.path.exists(DIRECTORIO_DATASET)
    print(f"La carpeta '{DIRECTORIO_DATASET}' existe y es accesible.")
except AssertionError:
    print(f"Error: No se encontró la carpeta '{DIRECTORIO_DATASET}'")

# Cargar datos de entrenamiento y validación
entrenamiento = datagen.flow_from_directory(
    DIRECTORIO_DATASET,
    target_size=(TAMANO_IMG, TAMANO_IMG),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validacion = datagen.flow_from_directory(
    DIRECTORIO_DATASET,
    target_size=(TAMANO_IMG, TAMANO_IMG),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Definir el modelo CNN
modeloCNN = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(TAMANO_IMG, TAMANO_IMG, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(6, activation='softmax')  # 6 clases: glass, paper, cardboard, plastic, metal, trash
])

# Compilar el modelo
modeloCNN.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Definir callback para TensorBoard
tensorboard = TensorBoard(log_dir='logs/cnn_trashnet')

# Entrenar el modelo
historial = modeloCNN.fit(
    entrenamiento,
    validation_data=validacion,
    epochs=EPOCHS,
    callbacks=[tensorboard],
    steps_per_epoch=len(entrenamiento),
    validation_steps=len(validacion)
)

# Guardar el modelo
modeloCNN.save('modelo_trashnet.h5')

# Evaluar el modelo
resultado = modeloCNN.evaluate(validacion)
print(f"Precisión del modelo: {resultado[1] * 100:.2f}%")

# Visualizar curvas de precisión y pérdida
def visualizar_resultados(historial):
    acc = historial.history['accuracy']
    val_acc = historial.history['val_accuracy']
    loss = historial.history['loss']
    val_loss = historial.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(12, 6))

    # Precisión
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Precisión de Entrenamiento')
    plt.plot(epochs_range, val_acc, label='Precisión de Validación')
    plt.legend(loc='lower right')
    plt.title('Precisión de Entrenamiento y Validación')

    # Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Pérdida de Entrenamiento')
    plt.plot(epochs_range, val_loss, label='Pérdida de Validación')
    plt.legend(loc='upper right')
    plt.title('Pérdida de Entrenamiento y Validación')

    plt.show()

# Llamar a la función para visualizar los resultados
visualizar_resultados(historial)