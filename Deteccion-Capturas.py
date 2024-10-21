import os
import cv2
import tkinter as tk
from tkinter import Label, Button, Toplevel, Canvas, Scrollbar, Frame
from PIL import Image, ImageTk
from io import BytesIO
import requests
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener la API Key desde la variable de entorno
api_key = os.getenv("PRIVATE_API_KEY")

# Verificar si la API Key se cargó correctamente
if api_key is None:
    raise ValueError("La API Key no se encontró. Asegúrate de que el archivo .env contiene PRIVATE_API_KEY correctamente.")

# URL del modelo en Roboflow
api_url = "https://detect.roboflow.com/10k/1"

def infer_image_from_roboflow(image):
    """Función para hacer la inferencia con la API de Roboflow en una imagen"""
    # Convertir la imagen a formato JPEG en memoria
    _, img_encoded = cv2.imencode('.jpg', image)
    img_bytes = img_encoded.tobytes()
    
    # Configurar el archivo como payload para la solicitud POST
    files = {
        'file': ('image.jpg', img_bytes, 'image/jpeg')
    }
    
    # Hacer la solicitud a la API de Roboflow
    response = requests.post(f"{api_url}?api_key={api_key}", files=files)
    
    # Procesar el resultado
    if response.status_code == 200:
        return response.json()  # Retornar el resultado en formato JSON
    else:
        return None  # Si hubo un error, retornar None

def map_class_name(class_name):
    """Función para mapear las clases originales a las nuevas etiquetas"""
    class_mapping = {
        'bottle': 'plástico',
        'can': 'metal',
        'glass': 'cristal',
        'paper': 'papel'
    }
    return class_mapping.get(class_name, class_name)  # Retorna la clase mapeada o la original si no está en el diccionario

class WasteSortingGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Sistema de Clasificación de Residuos")
        self.window.geometry("800x600")
        
        # Inicializar la cámara
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        # Frame para la visualización de la cámara
        self.camera_frame = Label(self.window)
        self.camera_frame.pack(pady=20)

        # Mejorar el estilo del botón "Enviar solicitud"
        self.send_button = Button(self.window, text="Enviar solicitud", command=self.send_request, 
                                  font=("Arial", 14), bg="black", fg="white", padx=10, pady=5, bd=3, relief="raised")
        self.send_button.pack(pady=20)

        self.update_frame()  # Comienza la videotransmisión de la cámara en la ventana

    def update_frame(self):
        """Función para capturar frames de la cámara y mostrar el video en tiempo real"""
        ret, frame = self.cap.read()
        if ret:
            # Convertir el frame de OpenCV a un formato compatible con Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            # Mostrar el frame en la ventana
            self.camera_frame.imgtk = imgtk
            self.camera_frame.configure(image=imgtk)

        # Repetir la actualización después de 30 ms para mantener el video en vivo
        self.window.after(30, self.update_frame)

    def send_request(self):
        """Función para tomar 3 capturas, enviarlas a la API, y mostrar las predicciones"""
        captures = []
        predictions = []
        for i in range(3):  # Capturar 3 frames
            ret, frame = self.cap.read()
            if ret:
                # Hacer la inferencia con Roboflow
                result = infer_image_from_roboflow(frame)

                # Guardar el resultado de la predicción
                if result and 'predictions' in result and isinstance(result['predictions'], list):
                    pred_classes = [f"{map_class_name(pred['class'])} ({pred['confidence'] * 100:.1f}%)" for pred in result['predictions']]
                    predictions.append(", ".join(pred_classes))
                else:
                    predictions.append("No se detectaron objetos.")

                # Redimensionar el frame antes de convertirlo
                frame_resized = cv2.resize(frame, (200, 150))  # Redimensionar a 200x150 píxeles
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)

                # Guardar la imagen procesada
                captures.append(imgtk)

        # Mostrar las imágenes y predicciones en una nueva ventana con scroll
        self.show_results_window(captures, predictions)

    def show_results_window(self, captures, predictions):
        """Función para mostrar los resultados en una nueva ventana con scroll y en formato horizontal"""
        results_window = Toplevel(self.window)
        results_window.title("Resultados de Clasificación")
        results_window.geometry("800x600")  # Tamaño de la ventana de resultados

        # Crear un contenedor con canvas y scrollbar
        canvas = Canvas(results_window)
        scrollbar = Scrollbar(results_window, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        # Vincular el evento de la rueda del mouse para habilitar el scroll
        canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1*(event.delta/120)), "units"))

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Título
        title_label = Label(scrollable_frame, text="Respuesta de solicitud", font=("Arial", 16))
        title_label.pack(pady=10)

        # Crear un contenedor horizontal para las imágenes y predicciones centrado
        row_frame = Frame(scrollable_frame)
        row_frame.pack(pady=10)

        # Mostrar las imágenes procesadas y las predicciones en horizontal
        for i, img in enumerate(captures):
            # Frame para cada imagen y predicción
            col_frame = Frame(row_frame)
            col_frame.pack(side="left", padx=30)  # Centramos al aumentar el padding entre las imágenes

            # Mostrar la imagen
            img_label = Label(col_frame, image=img)
            img_label.pack(pady=5)

            # Mostrar la predicción debajo de la imagen
            prediction_label = Label(col_frame, text=f"Captura {i+1}: {predictions[i]}", font=("Arial", 12))
            prediction_label.pack(pady=5)

            # Necesario para evitar que la imagen sea recolectada por el garbage collector
            img_label.image = img

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