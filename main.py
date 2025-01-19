from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Crear aplicación FastAPI
app = FastAPI()

# Configurar CORS para permitir solicitudes desde Netlify
origins = [
    "https://tu-sitio-netlify.netlify.app",  # Cambia esta URL por tu dominio en Netlify
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo para recibir datos
class Numbers(BaseModel):
    a: float
    b: float

# Ruta raíz para probar la API
@app.get("/")
def read_root():
    return {"message": "¡La API está funcionando correctamente!"}

# Ruta para sumar dos números
@app.post("/add")
def add_numbers(numbers: Numbers):
    result = numbers.a + numbers.b
    return {"result": result}
