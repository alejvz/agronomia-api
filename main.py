# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


app = FastAPI()

# Add the CORS middleware to allow requests from your React app's domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with the actual domain of your React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    pH: float
    MO: float
    P: float
    S: float
    Ca: float
    Mg: float
    Acidez: float
    Al: float
    K: float
    Na: float
    CE: float
    Fe: float
    Cu: float
    Mn: float
    Zn: float
    B: float

class ItemPh(BaseModel):
    CIC: float
    MO: float
    P: float
    S: float
    Ca: float
    Mg: float
    Acidez: float
    Al: float
    K: float
    Na: float
    CE: float
    Fe: float
    Cu: float
    Mn: float
    Zn: float
    B: float

class ItemMo(BaseModel):
    pH: float
    CIC: float
    P: float
    S: float
    Ca: float
    Mg: float
    Acidez: float
    Al: float
    K: float
    Na: float
    CE: float
    Fe: float
    Cu: float
    Mn: float
    Zn: float
    B: float

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/api/send-data")
async def send_data(item: Item):
    # Aquí puedes manejar los datos recibidos de tu frontend
    # En este ejemplo, simplemente los devolvemos como respuesta
    return item

@app.post("/api/predict")
async def predict(data: Item):
    print(data)
    poly_reg = PolynomialFeatures(degree=2)
    pkl_filename = "pickle_model_poly_reg.pkl"
    with open(pkl_filename, 'rb') as file:
        modelreg = pickle.load(file)
    data = np.array([[data.pH, data.MO, data.P, data.S, data.Ca, data.Mg, data.Acidez, data.Al, data.K, data.Na, data.CE, data.Fe, data.Cu, data.Mn, data.Zn, data.B]])
    data = poly_reg.fit_transform(data)
    prediction = modelreg.predict(data)
    return prediction[0]

@app.post("/api/ph")
async def predict(data: ItemPh):
    print(data)
    poly_reg = PolynomialFeatures(degree=2)
    pkl_filename = "pickle_model_poly_reg.pkl"
    with open(pkl_filename, 'rb') as file:
        modelreg = pickle.load(file)
    data = np.array([[data.CIC, data.MO, data.P, data.S, data.Ca, data.Mg, data.Acidez, data.Al, data.K, data.Na, data.CE, data.Fe, data.Cu, data.Mn, data.Zn, data.B]])
    data = poly_reg.fit_transform(data)
    prediction = modelreg.predict(data)
    return prediction[0]

@app.post("/api/mo")
async def predict(data: ItemMo):
    print(data)
    poly_reg = PolynomialFeatures(degree=2)
    pkl_filename = "pickle_model_poly_reg.pkl"
    with open(pkl_filename, 'rb') as file:
        modelreg = pickle.load(file)
    data = np.array([[data.CIC, data.pH, data.P, data.S, data.Ca, data.Mg, data.Acidez, data.Al, data.K, data.Na, data.CE, data.Fe, data.Cu, data.Mn, data.Zn, data.B]])
    data = poly_reg.fit_transform(data)
    prediction = modelreg.predict(data)
    return prediction[0]

