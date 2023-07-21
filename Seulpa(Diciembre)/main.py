from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

scaler = joblib.load('C:/Users/ruben/Documents/Proyecto Segulpa/modelos/scaler.pkl')

def model_prediction(x_in, model):
    
    x_in = np.asarray(x_in).reshape(1,-1)
    x_in = scaler.transform(x_in)

    return model.predict(x_in)

class Request(BaseModel):
    parking:int
    day: int
    month:int
    year:int
    hour:int
    school_day:int
    holiday:int
    temperatura:int
    humedad:int
    week_day : int
    model : str

'''def model_prediction(x_in, model):
    x = np.asarray(x_in).reshape(1,-1)
    preds=model.predict(x)

    return preds'''

# Configuración de CORS
origins = [
    "http://127.0.0.1:5500" # the origin of my live server, due to this is not an oficial task with professional achivements, i dont use env file
]


# inicialization of the middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
async def home(request: Request):
    if request.model == 'checkin':
        clf = joblib.load("C:/Users/ruben/Documents/Proyecto Segulpa/modelos/CheckIn.pkl")
    elif request.model == 'checkout':
        clf = joblib.load("C:/Users/ruben/Documents/Proyecto Segulpa/modelos/CheckOut.pkl")

    x_in =[np.float_(request.parking),
                    np.float_(request.day),
                    np.float_(request.month),
                    np.float_(request.year),
                    np.float_(request.hour),
                    np.float_(request.school_day),
                    np.float_(request.holiday),
                    np.float_(request.week_day),
                    np.float_(request.humedad),
                    np.float_(request.temperatura)]
    predictS = model_prediction(x_in, clf)
    return [int(predictS[0]), request.model]
    

