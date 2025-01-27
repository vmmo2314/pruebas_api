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
        "https://footcareai.netlify.app",  # Dominio de producción
        "https://conexioesp32.netlify.app"  # Dominio de germen
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos HTTP
    allow_headers=["*"],  # Permitir todos los encabezados
)

# Esquema de entrada para 14 características
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float
    feature6: float
    feature7: float
    feature8: float
    feature9: float
    feature10: float
    feature11: float
    feature12: float
    feature13: float
    feature14: float

# Ruta de prueba
@app.get("/")
def home():
    return {"message": "Bienvenido a la API de predicción"}

# Ruta para predicciones
@app.post("/predict")
def predict(data: InputData):
    # Convertir los datos de entrada en un array de numpy
    features = np.array([[
        data.feature1, data.feature2, data.feature3, data.feature4,
        data.feature5, data.feature6, data.feature7, data.feature8,
        data.feature9, data.feature10, data.feature11, data.feature12,
        data.feature13, data.feature14
    ]])
    
    # Realizar la predicción
    prediction = model.predict(features)
    return {"prediction": prediction[0]}
