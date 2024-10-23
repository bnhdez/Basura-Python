import os
import cv2
import tkinter as tk
from dotenv import load_dotenv  # Importar dotenv para cargar las variables de entorno
from tkinter import Label
from PIL import Image, ImageTk
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
    api_url="https://detect.roboflow.com",
    api_key=api_key
)

def infer_image_from_roboflow(image_path):
    """Función para hacer la inferencia con la API de Roboflow en una imagen"""
    result = CLIENT.infer(image_path, model_id="10k/1")
    return result

def map_class_name(class_name):
    """Función para mapear las clases originales a las nuevas etiquetas"""
    class_mapping = {
        'bottle': 'plástico',
        'can': 'metal',
        'glass': 'cristal',
        'paper': 'papel'
    }
    return class_mapping.get(class_name, class_name)  # Retorna la clase mapeada o la original si no está en el diccionario

def draw_detections(frame, predictions):
    """Función para dibujar las detecciones en el frame"""
    for detection in predictions:
        if 'x' in detection and 'y' in detection and 'width' in detection and 'height' in detection:
            x = detection['x']
            y = detection['y']
            width = detection['width']
            height = detection['height']
            confidence = detection['confidence']
            class_name = map_class_name(detection['class'])  # Mapeamos la clase detectada

            start_point = (int(x - width / 2), int(y - height / 2))
            end_point = (int(x + width / 2), int(y + height / 2))

            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

            label = f'{class_name} ({confidence * 100:.1f}%)'
            cv2.putText(frame, label, (start_point[0], start_point[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

class WasteSortingGUI:
    def __init__(self, window):  # Aquí está el constructor corregido
        self.window = window
        self.window.title("Sistema de Clasificación de Residuos")
        self.window.geometry("800x600")
        
        # Inicializar la cámara
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        # Inicializar contador de residuos
        self.organic_count = 0
        self.plastic_count = 0
        self.glass_count = 0
        
        # Frame para la visualización de la cámara
        self.camera_frame = Label(self.window)
        self.camera_frame.pack(pady=20)
        
        # Estadísticas de clasificación
        self.stats_label = Label(self.window, text="Estadísticas de clasificación", font=("Arial", 14))
        self.stats_label.pack()

        self.organic_label = Label(self.window, text=f"Residuos Orgánicos: {self.organic_count}", font=("Arial", 12))
        self.organic_label.pack()
        
        self.plastic_label = Label(self.window, text=f"Plásticos: {self.plastic_count}", font=("Arial", 12))
        self.plastic_label.pack()
        
        self.glass_label = Label(self.window, text=f"Cristales: {self.glass_count}", font=("Arial", 12))
        self.glass_label.pack()

        self.update_frame()  # Llamada inicial para empezar a mostrar la cámara

    def update_frame(self):
        """Función para capturar frames de la cámara y mostrar las detecciones en tiempo real"""
        ret, frame = self.cap.read()
        if ret:
            # Guardar el frame temporalmente para enviar a Roboflow
            image_path = "temp_frame.jpg"
            cv2.imwrite(image_path, frame)

            # Hacer la inferencia con Roboflow
            result = infer_image_from_roboflow(image_path)

            # Dibujar las detecciones en el frame
            if result and 'predictions' in result and isinstance(result['predictions'], list):
                draw_detections(frame, result['predictions'])
                # Actualizar estadísticas según el tipo de residuo detectado
                for pred in result['predictions']:
                    class_name = map_class_name(pred['class'])
                    if class_name == 'plástico':
                        self.plastic_count += 1
                    elif class_name == 'metal':
                        self.plastic_count += 1  # Puedes agregar un contador adicional para metal si es necesario
                    elif class_name == 'cristal':
                        self.glass_count += 1
                    elif class_name == 'papel':
                        self.organic_count += 1  # Asumí que el papel se contabiliza como orgánico, puedes modificar esto

            # Actualizar los valores en la interfaz
            self.organic_label.config(text=f"Residuos Orgánicos: {self.organic_count}")
            self.plastic_label.config(text=f"Plásticos: {self.plastic_count}")
            self.glass_label.config(text=f"Cristales: {self.glass_count}")

            # Convertir el frame de OpenCV a un formato compatible con Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            # Mostrar el frame en la ventana
            self.camera_frame.imgtk = imgtk
            self.camera_frame.configure(image=imgtk)

        # Repetir la actualización después de 30 ms
        self.window.after(30, self.update_frame)

    def on_closing(self):
        """Función para cerrar la aplicación y liberar la cámara"""
        self.cap.release()
        self.window.destroy()

# Crear la ventana principal de la interfaz
root = tk.Tk()
app = WasteSortingGUI(root)

# Configurar la acción para cerrar la ventana correctamente
root.protocol("WM_DELETE_WINDOW", app.on_closing)

# Iniciar el loop de la interfaz
root.mainloop()