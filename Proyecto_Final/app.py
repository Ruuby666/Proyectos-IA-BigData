from pydantic import BaseModel
from flask import Flask, render_template, request
import joblib
import numpy as np

# Crear una instancia de la aplicación Flask
app = Flask(__name__)

scaler = joblib.load(
    'C:/Users/ruben/Documents/proyectos_VS_ CEIABDTA/Proyecto_Final/modelos/scaler.pkl')
modelh1n1 = joblib.load(
    'C:/Users/ruben/Documents/proyectos_VS_ CEIABDTA/Proyecto_Final/modelos/h1n1.pkl')
modelseasonal = joblib.load(
    'C:/Users/ruben/Documents/proyectos_VS_ CEIABDTA/Proyecto_Final/modelos/seasonal.pkl')

    # Elegir las variables que van en cada uno
def model_prediction_h1n1(h1n1_in, model):
    h1n1_in = np.asarray(h1n1_in).reshape(1, -1)
    h1n1_in = scaler.transform(h1n1_in)
    return model.predict(h1n1_in)

def model_prediction_seasonal(seasonal_in, model):
    seasonal_in = np.asarray(seasonal_in).reshape(1, -1)
    seasonal_in = scaler.transform(seasonal_in)
    return model.predict(seasonal_in)

class Request(BaseModel):
    respuesta1: int
    respuesta2: int
    respuesta3: int
    respuesta4: int
    respuesta5: int
    respuesta6: int
    respuesta7: int
    respuesta8: int
    respuesta9: int
    respuesta10: int
    respuesta11: int
    respuesta12: int
    respuesta13: int
    respuesta14: int
    respuesta15: int
    respuesta16: int
    respuesta17: int
    respuesta18: int
    respuesta19: int
    respuesta20: int
    respuesta21: int
    respuesta22: int
    respuesta23: int
    respuesta24: int
    respuesta25: int
    respuesta26: int


@app.route('/', methods=['GET'])
def home():

    return render_template('index.html')

# Define a route that receives the input data and returns the prediction
@app.route ('/predict', methods=['POST'])
def predict ():
    # Get the input data from the form

    h1n1_in = [
        np.int_(request.form["respuesta1"]),
        np.int_(request.form["respuesta2"]),
        np.int_(request.form["respuesta3"]),
        np.int_(request.form["respuesta4"]),
        np.int_(request.form["respuesta5"]),
        np.int_(request.form["respuesta6"]),
        np.int_(request.form["respuesta8"]),
        np.int_(request.form["respuesta9"]),
        np.int_(request.form["respuesta10"]),
        np.int_(request.form["respuesta11"]),
        np.int_(request.form["respuesta12"]),
        np.int_(request.form["respuesta13"]),
        np.int_(request.form["respuesta17"]),
        np.int_(request.form["respuesta18"]),
        np.int_(request.form["respuesta19"]),
        np.int_(request.form["respuesta20"]),
        np.int_(request.form["respuesta21"]),
        np.int_(request.form["respuesta22"]),
        np.int_(request.form["respuesta23"]),
        np.int_(request.form["respuesta24"]),
        np.int_(request.form["respuesta25"]),
        np.int_(request.form["respuesta26"])]

    seasonal_in = [
        np.int_(request.form["respuesta1"]),
        np.int_(request.form["respuesta2"]),
        np.int_(request.form["respuesta3"]),
        np.int_(request.form["respuesta4"]),
        np.int_(request.form["respuesta5"]),
        np.int_(request.form["respuesta7"]),
        np.int_(request.form["respuesta8"]),
        np.int_(request.form["respuesta9"]),
        np.int_(request.form["respuesta10"]),
        np.int_(request.form["respuesta14"]),
        np.int_(request.form["respuesta15"]),
        np.int_(request.form["respuesta16"]),
        np.int_(request.form["respuesta17"]),
        np.int_(request.form["respuesta18"]),
        np.int_(request.form["respuesta19"]),
        np.int_(request.form["respuesta20"]),
        np.int_(request.form["respuesta21"]),
        np.int_(request.form["respuesta22"]),
        np.int_(request.form["respuesta23"]),
        np.int_(request.form["respuesta24"]),
        np.int_(request.form["respuesta25"]),
        np.int_(request.form["respuesta26"])]
    
    print(h1n1_in)
    print(seasonal_in)
    # Make a prediction using the model
    predictH = model_prediction_h1n1(h1n1_in, modelh1n1)
    predictS = model_prediction_seasonal(seasonal_in, modelseasonal)

    if (predictH == 0):
      sol1 = 'No'
    else:
      sol1 = 'Si'
    

    if (predictS == 0):
      sol2 = 'No'
    else:
      sol2 = 'Si'

    # Return the prediction as a string
    return render_template('index.html', resultados = True, sol1 = sol1, sol2 = sol2)

# Iniciar el servidor web de Flask
if __name__ == '__main__':
    app.run(debug=True)
