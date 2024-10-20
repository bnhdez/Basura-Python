import os
import cv2 
import numpy as np 
from dotenv import load_dotenv 
from inference_sdk import InferenceHTTPClient 

# Cargar las variables de entorno desde el archivo .env
load_dotenv()
api_key = os.getenv("PRIVATE_API_KEY")

# Verificar si la API Key se cargó correctamente
if api_key is None:
    raise ValueError("La API Key no se encontró. Asegúrate de que el archivo .env contiene PRIVATE_API_KEY correctamente.")

# Configurar el cliente de inferencia con la API de Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=api_key
)

# Función para hacer la inferencia en un frame de la cámara
def infer_frame(frame):
    # Guardar temporalmente el frame como imagen
    cv2.imwrite("temp_frame.jpg", frame)
    
    # Hacer la inferencia en la imagen temporal
    result = CLIENT.infer("temp_frame.jpg", model_id="waste-detection-ctmyy/9")
    
    return result

# Función para dibujar los cuadros de detección en el frame
def draw_detections(frame, detections):
    for detection in detections:
        x = detection['x']
        y = detection['y']
        width = detection['width']
        height = detection['height']
        confidence = detection['confidence']
        class_name = detection['class']

        # Calcular las coordenadas del cuadro
        start_point = (int(x - width / 2), int(y - height / 2))
        end_point = (int(x + width / 2), int(y + height / 2))

        # Dibujar el cuadro y el texto
        cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
        cv2.putText(frame, f'{class_name} ({confidence*100:.1f}%)', (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Inicializar la cámara
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)  # Fuerza el uso de DirectShow en lugar de MSMF
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)  # Ajusta el brillo
#cap.set(cv2.CAP_PROP_EXPOSURE, 0.1)  # Ajusta la exposición
if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

print("Presiona 'q' para cerrar la cámara.")

while True:
    # Leer un frame de la cámara
    ret, frame = cap.read()
    
    if not ret:
        print("No se pudo capturar el frame.")
        break

    # Hacer la inferencia en el frame actual
    result = infer_frame(frame)
    print(result)

    # Dibujar las detecciones en el frame
    draw_detections(frame, result['predictions'])

    # Mostrar el frame con las detecciones
    cv2.imshow('Detección de basura en tiempo real', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()