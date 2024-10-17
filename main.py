import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener la API Key de las variables de entorno
api_key = os.getenv("PRIVATE_API_KEY")

# Verificar si la API Key se cargó correctamente
print(f"API Key cargada: {api_key}")  # Línea de depuración
if api_key is None:
    raise ValueError("La API Key no se encontró. Asegúrate de que el archivo .env contiene PRIVATE_API_KEY correctamente.")

# Configurar el cliente de inferencia con la API de Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=api_key
)

# Hacer la inferencia en una imagen de prueba
result = CLIENT.infer("prueba.jpg", model_id="waste-detection-ctmyy/9")
print(result)