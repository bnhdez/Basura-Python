import os
import cv2
from inference_sdk import InferenceHTTPClient
#CLASIFICADOR DE BASURA
# Cargar la API Key directamente
api_key = "zbnsnKkZjuLCzZhPrRAc"  # Asegúrate de que esta es tu API Key válida

# Inicializar el cliente de inferencia con Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://classify.roboflow.com",
    api_key=api_key
)

def infer_image_from_roboflow(image_path):
    """Función para hacer la inferencia con la API de Roboflow en una imagen"""
    result = CLIENT.infer(image_path, model_id="recyclingman/3")
    return result

def draw_classification(frame, predictions):
    """Función para dibujar la clase predicha en el frame"""
    # Encontrar la clase con la mayor confianza
    best_class = None
    best_confidence = 0.0
    
    for class_name, details in predictions.items():
        confidence = details['confidence']
        if confidence > best_confidence:
            best_confidence = confidence
            best_class = class_name

    # Mostrar la clase predicha con la mayor confianza
    if best_class is not None:
        label = f'{best_class} ({best_confidence * 100:.1f}%)'
        cv2.putText(frame, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Capturar la cámara en tiempo real desde el índice 2
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)  # Asegúrate de usar DirectShow en Windows para mayor compatibilidad

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

    # Dibujar la clase predicha en el frame
    if result and 'predictions' in result:
        draw_classification(frame, result['predictions'])

    # Mostrar el frame en una ventana de OpenCV
    cv2.imshow('Captura de residuos en tiempo real', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()






