import os
import cv2
import tkinter as tk
from tkinter import Label, Button, Toplevel, Frame, PhotoImage
from PIL import Image, ImageTk
import requests
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog
from dotenv import load_dotenv
import pyodbc
from datetime import datetime

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener la API Key desde el archivo .env
api_key = os.getenv("PRIVATE_API_KEY")

if api_key is None:
    raise ValueError("La API Key no se encontró. Asegúrate de que el archivo .env contiene PRIVATE_API_KEY correctamente.")

# URL del modelo en Roboflow
api_url = "https://detect.roboflow.com/10k/1"

# Diccionario para contar los tipos de residuos
residuo_contador = {'plástico': 0, 'vidrio': 0, 'metal': 0, 'papel': 0, 'otros': 0}

# Conexión a SQL Server
conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};'
                      'SERVER=localhost;'
                      'DATABASE=WasteSortingDB;'
                      'Trusted_Connection=yes;')
cursor = conn.cursor()

# Crear tabla en SQL Server si no existe
cursor.execute('''
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='waste_stats' AND xtype='U')
CREATE TABLE waste_stats (
    id INT PRIMARY KEY IDENTITY(1,1),
    plastic_count INT,
    glass_count INT,
    metal_count INT,
    paper_count INT,
    others_count INT,
    timestamp DATETIME DEFAULT GETDATE()
)
''')
conn.commit()

class WasteSortingGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Sistema de clasificación de residuos")

        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        self.window.geometry(f"{screen_width}x{screen_height}")
        self.window.configure(bg='#ffffff')

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

        self.main_frame = Frame(self.window, bg='#ffffff')
        self.main_frame.pack(fill="both", expand=True, pady=10)

        camera_frame_container = Frame(self.main_frame, bg='#ffffff')
        camera_frame_container.pack(pady=10)

        self.camera_frame = Label(camera_frame_container, bg='#000')
        self.camera_frame.pack()

        self.send_button = Button(self.main_frame, text="ENVIAR SOLICITUD", command=self.send_request,
                                  font=("Arial", 16, "bold"), bg="#b3f35a", fg="black", padx=20, pady=10, bd=0, relief="flat")
        self.send_button.pack(pady=5)

        self.stats_button = Button(self.main_frame, text="VER ESTADÍSTICAS", command=self.show_stats,
                                   font=("Arial", 16, "bold"), bg="#b3f35a", fg="black", padx=20, pady=10, bd=0, relief="flat")
        self.stats_button.pack(pady=5)

        self.history_button = Button(self.main_frame, text="HISTORICO", command=self.show_history,
                                     font=("Arial", 16, "bold"), bg="#b3f35a", fg="black", padx=20, pady=10, bd=0, relief="flat")
        self.history_button.pack(pady=5)

        self.cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

        self.window_closed = False
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_frame.imgtk = imgtk
            self.camera_frame.configure(image=imgtk)

        if not self.window_closed:
            self.window.after(30, self.update_frame)

    def send_request(self):
        self.captures = []
        self.predictions = []
        self.capture_images = []

        for i in range(3):
            ret, frame = self.cap.read()
            if ret:
                result = self.infer_image_from_roboflow(frame)
                if result and 'predictions' in result and isinstance(result['predictions'], list):
                    pred_classes = []
                    for pred in result['predictions']:
                        mapped_class = self.map_class_name(pred['class'])
                        pred_classes.append(f"{mapped_class} ({pred['confidence'] * 100:.1f}%)")
                        residuo_contador[mapped_class] += 1
                    self.predictions.append(", ".join(pred_classes))
                    frame = self.draw_boxes_on_frame(frame, result['predictions'])
                else:
                    self.predictions.append("No se detectaron objetos.")
                    residuo_contador['otros'] += 1

                frame_resized = cv2.resize(frame, (200, 150))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.captures.append(imgtk)
                self.capture_images.append(img)

        self.save_to_database()
        self.show_results_window()

    def save_to_database(self):
        cursor.execute('''
            INSERT INTO waste_stats (plastic_count, glass_count, metal_count, paper_count, others_count)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            residuo_contador['plástico'],
            residuo_contador['vidrio'],
            residuo_contador['metal'],
            residuo_contador['papel'],
            residuo_contador['otros']
        ))
        conn.commit()

    def infer_image_from_roboflow(self, image):
        _, img_encoded = cv2.imencode('.jpg', image)
        img_bytes = img_encoded.tobytes()
        files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}
        response = requests.post(f"{api_url}?api_key={api_key}", files=files)
        return response.json() if response.status_code == 200 else None

    def map_class_name(self, class_name):
        class_mapping = {'bottle': 'plástico', 'can': 'metal', 'glass': 'vidrio', 'paper': 'papel'}
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

    def show_history(self):
        self.window.withdraw()
        self.show_history_window()

    def show_history_window(self):
        history_window = Toplevel(self.window)
        history_window.title("Histórico")
        history_window.geometry("900x650")
        history_window.configure(bg='#ffffff')

        header_frame = Frame(history_window, bg='#a0e75a', height=80)
        header_frame.pack(fill="x")
        title_label = Label(header_frame, text="HISTORICO", font=("Arial", 20, "bold"), bg='#a0e75a', fg="black")
        title_label.pack(pady=10)

        from_label = Label(history_window, text="Desde", font=("Arial", 14), bg='#ffffff', fg="black")
        from_label.pack(pady=5)
        from_entry = tk.Entry(history_window, font=("Arial", 14), width=10)
        from_entry.insert(0, "DD/MM/YYYY")
        from_entry.pack(pady=5)

        to_label = Label(history_window, text="Hasta", font=("Arial", 14), bg='#ffffff', fg="black")
        to_label.pack(pady=5)
        to_entry = tk.Entry(history_window, font=("Arial", 14), width=10)
        to_entry.insert(0, "DD/MM/YYYY")
        to_entry.pack(pady=5)

        search_button = Button(history_window, text="Buscar", font=("Arial", 12), bg="#b3f35a", fg="black", command=lambda: self.search_history(from_entry.get(), to_entry.get(), history_window))
        search_button.pack(pady=10)

        self.table_frame = Frame(history_window, bg='#ffffff')
        self.table_frame.pack(pady=20)

        Button(history_window, text="REGRESAR", font=("Arial", 12), bg="#b3f35a", fg="black", bd=0, command=lambda: self.back_to_main(history_window)).pack(pady=20)

    def back_to_main(self, history_window):
        history_window.destroy()
        self.window.deiconify()

    def search_history(self, from_date, to_date, history_window):
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        try:
            from_date_obj = datetime.strptime(from_date, "%d/%m/%Y")
            to_date_obj = datetime.strptime(to_date, "%d/%m/%Y")
        except ValueError:
            print("Formato de fecha incorrecto. Usa DD/MM/YYYY.")
            return

        if from_date_obj == to_date_obj:
            query = '''
                SELECT plastic_count, glass_count, paper_count, metal_count, others_count, timestamp 
                FROM waste_stats
                WHERE CONVERT(date, timestamp) = ?
                ORDER BY timestamp
            '''
            cursor.execute(query, from_date_obj)
        else:
            query = '''
                SELECT plastic_count, glass_count, paper_count, metal_count, others_count, timestamp 
                FROM waste_stats
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            '''
            cursor.execute(query, from_date_obj, to_date_obj)

        rows = cursor.fetchall()

        headers = ['Plástico', 'Vidrio', 'Papel', 'Metal', 'Otros', 'Fecha']
        for col, header in enumerate(headers):
            header_label = Label(self.table_frame, text=header, font=("Arial", 14, "bold"), bg='#ffffff', fg="black", relief="solid", bd=1, width=15)
            header_label.grid(row=0, column=col)

        for row_num, row in enumerate(rows, start=1):
            for col_num, value in enumerate(row[:-1]):
                cell_label = Label(self.table_frame, text=value, font=("Arial", 12), bg='#ffffff', fg="black", relief="solid", bd=1, width=20)
                cell_label.grid(row=row_num, column=col_num)
            date_label = Label(self.table_frame, text=row[-1].strftime('%Y-%m-%d %H:%M'), font=("Arial", 12), bg='#ffffff', fg="black", relief="solid", bd=1, width=20)
            date_label.grid(row=row_num, column=5)

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

            Button(col_frame, text="DESCARGAR", font=("Arial", 12), bg="#b3f35a", fg="black", bd=0, command=lambda i=i: self.download_image(i)).pack(pady=5)

        Button(results_window, text="DESCARGAR TODO", font=("Arial", 12), bg="#b3f35a", fg="black", bd=0, command=self.download_all).pack(pady=20)

    def download_image(self, index):
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
        if file_path:
            self.capture_images[index].save(file_path)

    def download_all(self):
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
        self.window_closed = True
        self.cap.release()
        conn.close()
        self.window.destroy()

# Crear la ventana principal
root = tk.Tk()
app = WasteSortingGUI(root)

root.protocol("WM_DELETE_WINDOW", app.on_closing)
root.mainloop()