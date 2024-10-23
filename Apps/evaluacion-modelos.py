import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Definir parámetros
TAMANO_IMG = 224  # Tamaño de las imágenes ajustado para coincidir con el modelo
BATCH_SIZE = 32   # Tamaño del lote para entrenamiento

# Directorios de los datasets
DIRECTORIO_TRASHNET = './dataset-original'
DIRECTORIO_GARBAGE = './Garbage classification'
DIRECTORIO_TACO = './TACO'

# Configurar generadores de datos
datagen = ImageDataGenerator(
    rescale=1./255,        # Normalizar imágenes
    validation_split=0.15   # Separar 15% para validación
)

# Cargar datos de validación para cada dataset con el tamaño correcto
validacion_trashnet = datagen.flow_from_directory(
    DIRECTORIO_TRASHNET,
    target_size=(TAMANO_IMG, TAMANO_IMG),  # Tamaño ajustado
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

validacion_garbage = datagen.flow_from_directory(
    DIRECTORIO_GARBAGE,
    target_size=(TAMANO_IMG, TAMANO_IMG),  # Tamaño ajustado
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

validacion_taco = datagen.flow_from_directory(
    DIRECTORIO_TACO,
    target_size=(TAMANO_IMG, TAMANO_IMG),  # Tamaño ajustado
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Función para convertir DirectoryIterator a tf.data.Dataset
def convertir_a_dataset(directory_iterator):
    def generator():
        for batch in directory_iterator:
            yield batch[0], batch[1]  # X (imágenes) e Y (etiquetas)
    return tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(None, TAMANO_IMG, TAMANO_IMG, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 6), dtype=tf.float32)))

# Convertir cada DirectoryIterator a tf.data.Dataset
ds_validacion_trashnet = convertir_a_dataset(validacion_trashnet)
ds_validacion_garbage = convertir_a_dataset(validacion_garbage)
ds_validacion_taco = convertir_a_dataset(validacion_taco)

# Combinar los datasets de validación
validacion_combined = ds_validacion_trashnet.concatenate(ds_validacion_garbage).concatenate(ds_validacion_taco)

# Cargar el modelo guardado
modelo_guardado = load_model('modelo_mejorado.h5')

# Evaluar el modelo en el conjunto de validación combinado
resultado = modelo_guardado.evaluate(validacion_combined)
print(f"Precisión del modelo guardado: {resultado[1] * 100:.2f}%")