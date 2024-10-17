import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Cargar el modelo guardado
modelo = tf.keras.models.load_model('modelo_mejorado.h5')

# Cargar una nueva imagen para hacer predicciones
ruta_imagen = './prueba2.jpg'  # Cambia esto por la ruta de tu imagen
TAMANO_IMG = 224  # Tamaño de imagen al que el modelo fue entrenado

# Preprocesar la imagen
img = image.load_img(ruta_imagen, target_size=(TAMANO_IMG, TAMANO_IMG))
img_array = image.img_to_array(img) / 255.0  # Normalizar
img_array = np.expand_dims(img_array, axis=0)  # Expandir dimensiones para hacerla compatible con el modelo

# Hacer la predicción
prediccion = modelo.predict(img_array)

# Mostrar el resultado
clases = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
clase_predicha = np.argmax(prediccion)
print(f"El objeto es: {clases[clase_predicha]}")