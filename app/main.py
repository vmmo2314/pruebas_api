from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Inicializar FastAPI
app = FastAPI()

# Cargar el modelo entrenado
with open("app/model.pkl", "rb") as f:
    model = pickle.load(f)

# Definir el esquema de datos de entrada
class InputData(BaseModel):
    age: int
    income: float

# Ruta de prueba
@app.get("/")
def home():
    return {"message": "¡API de modelo predictivo funcionando!"}

# Ruta para realizar predicciones
@app.post("/predict/")
def predict(data: InputData):
    # Convertir datos a formato compatible con el modelo
    features = [[data.age, data.income]]
    # Realizar la predicción
    prediction = model.predict(features)[0]
    # Retornar la predicción
    return {"prediction": "Compra" if prediction == 1 else "No compra"}
