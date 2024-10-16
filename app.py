import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

# Definir parámetros
TAMANO_IMG = 100  # Tamaño de las imágenes
BATCH_SIZE = 32   # Tamaño del lote para entrenamiento
EPOCHS = 30       # Número de épocas de entrenamiento

# Directorios de los datasets
DIRECTORIO_TRASHNET = './dataset-original'
DIRECTORIO_GARBAGE = './Garbage classification'
DIRECTORIO_TACO = './TACO'

# Configurar generadores de datos
datagen = ImageDataGenerator(
    rescale=1./255,        # Normalizar imágenes
    validation_split=0.15   # Separar 15% para validación
)

# Convertir DirectoryIterator a tf.data.Dataset
def convertir_a_dataset(directory_iterator):
    def generator():
        for batch in directory_iterator:
            yield batch[0], batch[1]  # X (imágenes) e Y (etiquetas)
    return tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(None, TAMANO_IMG, TAMANO_IMG, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 6), dtype=tf.float32)))

# Cargar datos de entrenamiento y validación para cada dataset
entrenamiento_trashnet = datagen.flow_from_directory(
    DIRECTORIO_TRASHNET,
    target_size=(TAMANO_IMG, TAMANO_IMG),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validacion_trashnet = datagen.flow_from_directory(
    DIRECTORIO_TRASHNET,
    target_size=(TAMANO_IMG, TAMANO_IMG),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

entrenamiento_garbage = datagen.flow_from_directory(
    DIRECTORIO_GARBAGE,
    target_size=(TAMANO_IMG, TAMANO_IMG),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validacion_garbage = datagen.flow_from_directory(
    DIRECTORIO_GARBAGE,
    target_size=(TAMANO_IMG, TAMANO_IMG),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

entrenamiento_taco = datagen.flow_from_directory(
    DIRECTORIO_TACO,
    target_size=(TAMANO_IMG, TAMANO_IMG),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validacion_taco = datagen.flow_from_directory(
    DIRECTORIO_TACO,
    target_size=(TAMANO_IMG, TAMANO_IMG),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Convertir cada DirectoryIterator a tf.data.Dataset
ds_entrenamiento_trashnet = convertir_a_dataset(entrenamiento_trashnet)
ds_validacion_trashnet = convertir_a_dataset(validacion_trashnet)

ds_entrenamiento_garbage = convertir_a_dataset(entrenamiento_garbage)
ds_validacion_garbage = convertir_a_dataset(validacion_garbage)

ds_entrenamiento_taco = convertir_a_dataset(entrenamiento_taco)
ds_validacion_taco = convertir_a_dataset(validacion_taco)

# Combinar los datasets
entrenamiento_combined = ds_entrenamiento_trashnet.concatenate(ds_entrenamiento_garbage).concatenate(ds_entrenamiento_taco)
validacion_combined = ds_validacion_trashnet.concatenate(ds_validacion_garbage).concatenate(ds_validacion_taco)

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

# Callbacks: Guardar el modelo y parar si no mejora
tensorboard = TensorBoard(log_dir='logs/cnn_trashnet')
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
checkpointer = ModelCheckpoint(filepath='modelo_trashnet.h5', save_best_only=True, monitor='val_loss')

# Entrenar el modelo con los datasets combinados
historial = modeloCNN.fit(
    entrenamiento_combined,
    validation_data=validacion_combined,
    epochs=EPOCHS,
    callbacks=[tensorboard, early_stopping, checkpointer],
    steps_per_epoch=len(entrenamiento_trashnet),
    validation_steps=len(validacion_trashnet)
)

# Evaluar el modelo
resultado = modeloCNN.evaluate(validacion_combined)
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