import os
import cv2
import tkinter as tk
from tkinter import Label, Button, Toplevel, Frame, PhotoImage
from PIL import Image, ImageTk
from io import BytesIO
import requests
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog
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

# Diccionario para contar los tipos de residuos
residuo_contador = {
    'plástico': 0,
    'vidrio': 0,
    'metal': 0,
    'papel': 0,
    'otros': 0
}

class WasteSortingGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Sistema de clasificación de residuos")

        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        self.window.geometry(f"{screen_width}x{screen_height}")
        self.window.configure(bg='#ffffff')

        # Crear el encabezado con el logo y el título
        header_frame = Frame(self.window, bg='#a0e75a', height=80)
        header_frame.pack(fill="x")

        img = Image.open("recycle_icon.png")
        img_resized = img.resize((50, 50), Image.LANCZOS)
        logo = ImageTk.PhotoImage(img_resized)
        
        logo_label = Label(header_frame, image=logo, bg='#a0e75a')
        logo_label.image = logo
        logo_label.pack(side="left", padx=10)

        title_label = Label(header_frame, text="Sistema de clasificación de residuos", font=("Arial", 20, "bold"), bg='#a0e75a', fg="black")
        title_label.pack(side="left", padx=20)

        # Frame principal con fondo blanco
        self.main_frame = Frame(self.window, bg='#ffffff')
        self.main_frame.pack(fill="both", expand=True, pady=10)

        # Frame para la cámara
        camera_frame_container = Frame(self.main_frame, bg='#ffffff')
        camera_frame_container.pack(pady=10)
        
        self.camera_frame = Label(camera_frame_container, bg='#000')
        self.camera_frame.pack()

        # Botón "Enviar solicitud"
        self.send_button = Button(self.main_frame, text="ENVIAR SOLICITUD", command=self.send_request,
                                  font=("Arial", 16, "bold"), bg="#b3f35a", fg="black", padx=20, pady=10, bd=0, relief="flat")
        self.send_button.pack(pady=5)

        # Botón para ver estadísticas
        self.stats_button = Button(self.main_frame, text="VER ESTADÍSTICAS", command=self.show_stats,
                                   font=("Arial", 16, "bold"), bg="#b3f35a", fg="black", padx=20, pady=10, bd=0, relief="flat")
        self.stats_button.pack(pady=5)

        # Inicializa la cámara
        self.cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
        
        # Estado para controlar si la ventana está cerrada
        self.window_closed = False

        # Inicia el ciclo de actualización del frame de la cámara
        self.update_frame()

    def update_frame(self):
        """Función para actualizar el frame de la cámara"""
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_frame.imgtk = imgtk
            self.camera_frame.configure(image=imgtk)

        if not self.window_closed:
            # Programar la siguiente actualización si la ventana no está cerrada
            self.window.after(30, self.update_frame)

    def send_request(self):
        self.captures = []
        self.predictions = []
        self.capture_images = []  # Lista para almacenar las imágenes reales de PIL

        for i in range(3):  
            ret, frame = self.cap.read()
            if ret:
                result = self.infer_image_from_roboflow(frame)

                if result and 'predictions' in result and isinstance(result['predictions'], list):
                    pred_classes = []
                    for pred in result['predictions']:
                        mapped_class = self.map_class_name(pred['class'])
                        pred_classes.append(f"{mapped_class} ({pred['confidence'] * 100:.1f}%)")
                        # Actualizar el contador para la clase mapeada
                        residuo_contador[mapped_class] += 1
                    self.predictions.append(", ".join(pred_classes))
                    frame = self.draw_boxes_on_frame(frame, result['predictions'])
                else:
                    self.predictions.append("No se detectaron objetos.")
                    residuo_contador['otros'] += 1  # Si no se detectó, cuenta como "otros"

                frame_resized = cv2.resize(frame, (200, 150))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.captures.append(imgtk)
                self.capture_images.append(img)  # Guardar la imagen real de PIL

        self.show_results_window()

    def infer_image_from_roboflow(self, image):
        _, img_encoded = cv2.imencode('.jpg', image)
        img_bytes = img_encoded.tobytes()

        files = {
            'file': ('image.jpg', img_bytes, 'image/jpeg')
        }

        response = requests.post(f"{api_url}?api_key={api_key}", files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None

    def map_class_name(self, class_name):
        class_mapping = {
            'bottle': 'plástico',
            'can': 'metal',
            'glass': 'vidrio',
            'paper': 'papel'
        }
        return class_mapping.get(class_name, 'otros')

    def draw_boxes_on_frame(self, frame, predictions):
        for pred in predictions:
            if 'x' in pred and 'y' in pred and 'width' in pred and 'height' in pred:
                x = int(pred['x'] - pred['width'] / 2)
                y = int(pred['y'] - pred['height'] / 2)
                width = int(pred['width'])
                height = int(pred['height'])

                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        return frame

    def show_results_window(self):
        results_window = Toplevel(self.window)
        results_window.title("Resultados de Clasificación")
        results_window.geometry("900x650")
        results_window.configure(bg='#ffffff')

        header_frame = Frame(results_window, bg='#a0e75a', height=80)
        header_frame.pack(fill="x")
        title_label = Label(header_frame, text="CAPTURAS", font=("Arial", 20, "bold"), bg='#a0e75a', fg="black")
        title_label.pack(pady=10)

        row_frame = Frame(results_window, bg='#ffffff')
        row_frame.pack(pady=10)

        for i, img in enumerate(self.captures):
            col_frame = Frame(row_frame, bg='#ffffff')
            col_frame.pack(side="left", padx=30)

            img_label = Label(col_frame, image=img, bg='#ffffff')
            img_label.pack(pady=5)

            prediction_label = Label(col_frame, text=f"Captura {i+1}: {self.predictions[i]}", font=("Arial", 14), bg='#ffffff', fg="black")
            prediction_label.pack(pady=5)

            img_label.image = img

            # Botón para descargar una imagen individual
            Button(col_frame, text="DESCARGAR", font=("Arial", 12), bg="#b3f35a", fg="black", bd=0,
                   command=lambda i=i: self.download_image(i)).pack(pady=5)

        # Botón para descargar todas las imágenes
        Button(results_window, text="DESCARGAR TODO", font=("Arial", 12), bg="#b3f35a", fg="black", bd=0, command=self.download_all).pack(pady=20)

    def download_image(self, index):
        """Función para descargar una imagen individual"""
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
        if file_path:
            self.capture_images[index].save(file_path)

    def download_all(self):
        """Función para descargar todas las imágenes"""
        directory = filedialog.askdirectory()
        if directory:
            for i, img in enumerate(self.capture_images):
                file_path = os.path.join(directory, f"captura_{i+1}.jpg")
                img.save(file_path)

    def show_stats(self):
        stats_window = Toplevel(self.window)
        stats_window.title("Estadísticas")
        stats_window.geometry("900x650")
        stats_window.configure(bg='#ffffff')

        header_frame = Frame(stats_window, bg='#a0e75a', height=80)
        header_frame.pack(fill="x")
        title_label = Label(header_frame, text="ESTADÍSTICAS", font=("Arial", 20, "bold"), bg='#a0e75a', fg="black")
        title_label.pack(pady=10)

        fig, ax = plt.subplots(figsize=(6, 4))

        # Usar los datos del diccionario de residuos
        residuos = list(residuo_contador.keys())
        cantidades = list(residuo_contador.values())

        ax.bar(residuos, cantidades, color=['#4CAF50', '#FF5722', '#2196F3', '#FFC107', '#9E9E9E'])

        for i, v in enumerate(cantidades):
            ax.text(i, v + 10, str(v), ha='center', fontweight='bold')

        ax.set_title('Residuos capturados')
        ax.set_ylabel('Unidades')

        canvas = Frame(stats_window)
        canvas.pack()
        plt_canvas = FigureCanvasTkAgg(fig, master=canvas)
        plt_canvas.draw()
        plt_canvas.get_tk_widget().pack()

        Button(stats_window, text="REGRESAR", font=("Arial", 12), bg="#b3f35a", fg="black", bd=0, command=stats_window.destroy).pack(pady=20)

    def on_closing(self):
        """Manejo adecuado del cierre de la ventana"""
        self.window_closed = True  # Marcar la ventana como cerrada
        self.cap.release()  # Liberar la cámara
        self.window.destroy()

# Crear la ventana principal
root = tk.Tk()
app = WasteSortingGUI(root)

root.protocol("WM_DELETE_WINDOW", app.on_closing)
root.mainloop()
