from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

# Cargar el modelo
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8888",  # Para desarrollo
        "https://footcareai.netlify.app"  # Dominio de producción
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos HTTP
    allow_headers=["*"],  # Permitir todos los encabezados
)

# Esquema de entrada
class InputData(BaseModel):
    feature1: float
    feature2: float

# Ruta de prueba
@app.get("/")
def home():
    return {"message": "Bienvenido a la API de predicción"}

# Ruta para predicciones
@app.post("/predict")
def predict(data: InputData):
    features = np.array([[data.feature1, data.feature2]])
    prediction = model.predict(features)
    return {"prediction": prediction[0]}
