from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from cachetools import TTLCache
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pickle
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Cargar variables de entorno
load_dotenv()

# Configurar conexión con Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Cargar el modelo entrenado
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Crear el objeto FasAPI
app = FastAPI()

# Agregar CORS para permitir solicitudes del frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite cualquier origen, puedes restringirlo a ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Permite todos los encabezados
)

# Configuración de caché y concurrencia
cache = TTLCache(maxsize=100, ttl=60)
semaphore = asyncio.Semaphore(5)
executor = ThreadPoolExecutor()

# Esquema de entrada
class InputData(BaseModel):
    features: list[int]  # Asegurar que las características sean enteros

@ app.post("/predict")
async def predict(data: InputData):
    async with semaphore:
        features = tuple(data.features)

        if len(features) != 14:
            return {"error": "Se requieren exactamente 14 características"}

        if features in cache:
            prediction_value = cache[features]
            cached = True
        else:
            features_array = np.array([data.features])
            prediction = await asyncio.get_event_loop().run_in_executor(
                executor, model.predict, features_array
            )
            prediction_value = int(round(prediction[0]))
            cache[features] = prediction_value
            cached = False
            # Guardar en Supabase
            data_to_store = {
                "features": data.features,
                "prediction": prediction_value,
            }
            supabase.table("predictions").insert(data_to_store).execute()

        return {"prediction": prediction_value, "Cache": str(cached), "DB_Status": "Success"}