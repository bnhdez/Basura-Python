import os
import cv2
from dotenv import load_dotenv  # Importar dotenv para cargar las variables de entorno
from inference_sdk import InferenceHTTPClient

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener la API Key desde la variable de entorno
api_key = os.getenv("PRIVATE_API_KEY")

# Verificar si la API Key se cargó correctamente
if api_key is None:
    raise ValueError("La API Key no se encontró. Asegúrate de que el archivo .env contiene PRIVATE_API_KEY correctamente.")

# Inicializar el cliente de inferencia con Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",  # URL para detección
    api_key=api_key
)

def infer_image_from_roboflow(image_path):
    """Función para hacer la inferencia con la API de Roboflow en una imagen"""
    # Cambia el model_id para el modelo de detección que mencionaste
    result = CLIENT.infer(image_path, model_id="garbage-classification-3/2")
    return result

def draw_detections(frame, predictions):
    """Función para dibujar los cuadros de detección en el frame"""
    for detection in predictions:
        # Extraer coordenadas y detalles de las predicciones
        if 'x' in detection and 'y' in detection and 'width' in detection and 'height' in detection:
            x = detection['x']
            y = detection['y']
            width = detection['width']
            height = detection['height']
            confidence = detection['confidence']
            class_name = detection['class']

            # Calcular las coordenadas del cuadro delimitador
            start_point = (int(x - width / 2), int(y - height / 2))
            end_point = (int(x + width / 2), int(y + height / 2))

            # Dibujar el cuadro en el frame
            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

            # Mostrar el nombre de la clase y la confianza
            label = f'{class_name} ({confidence * 100:.1f}%)'
            cv2.putText(frame, label, (start_point[0], start_point[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Capturar la cámara en tiempo real desde el índice 2
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Asegúrate de usar DirectShow en Windows para mayor compatibilidad

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()  # Leer un frame de la cámara
    if not ret:
        print("Error al capturar el frame.")
        break

    # Guardar el frame como imagen temporal
    image_path = "temp_frame.jpg"
    cv2.imwrite(image_path, frame)

    # Hacer la inferencia con la API de Roboflow
    result = infer_image_from_roboflow(image_path)

    # Imprimir el resultado para verificar la estructura
    print("Resultado devuelto por Roboflow:")
    print(result)

    # Dibujar las detecciones en el frame si hay predicciones
    if result and 'predictions' in result and isinstance(result['predictions'], list):
        draw_detections(frame, result['predictions'])

    # Mostrar el frame en una ventana de OpenCV
    cv2.imshow('Captura de residuos en tiempo real', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()
