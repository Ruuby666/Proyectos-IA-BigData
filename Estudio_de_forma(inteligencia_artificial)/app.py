import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import joblib
import streamlit as st



# Path del modelo preentrenado
MODEL_PATH = 'models/Estudio_fisico.pkl'

# Se reciben los valores y el modelo, devuelve la predicción
def model_prediction(x_in, model):

    x = np.asarray(x_in).reshape(1,-1)
    preds=model.predict(x)

    return preds


def main():
    
    model=''

    # Se carga el modelo
    if model=='':
        with open(MODEL_PATH, 'rb') as file:
            model = joblib.load(file)
    
    # Título
    html_temp = """
    <h1 style="color:#181082;text-align:center;">Estudio del Esfuerzo realizado</h1>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    html_col = """
    <h5 style="color:#181082;text-align:left;">Trail running , Carrera </h5>
    """
    st.markdown(html_col,unsafe_allow_html=True)
    html_cla = """
    <h5 style="color:#181082;text-align:left;">0, 1</h5>
    """
    st.markdown(html_cla,unsafe_allow_html=True)

    # Lecctura de datos
    actividad = st.text_input("Tipo de actividad :")
    distancia = st.text_input("Distancia:")
    calorias = st.text_input("Calorías:")
    aerobico = st.text_input("TE aeróbico:")
    carreramedia = st.text_input("Cadencia de carrera media:")
    ritmo = st.text_input("Ritmo medio:")
    ascenso = st.text_input("Ascenso total:")
    descenso = st.text_input("Descenso total:")
    zancada = st.text_input("Longitud media de zancada:")
    tiempo = st.text_input("Tiempo en segundos:")
    temperatura = st.text_input("Temperatura media:")
    altura = st.text_input("Altura media:")
    
    

    # El botón predicción se usa para iniciar el procesamiento
    if st.button("Predicción :"):         
        x_in =[np.float_(actividad.title()),
                    np.float_(distancia.title()),
                    np.float_(calorias.title()),
                    np.float_(aerobico.title()),
                    np.float_(carreramedia.title()),
                    np.float_(ritmo.title()),
                    np.float_(ascenso.title()),
                    np.float_(descenso.title()),
                    np.float_(zancada.title()),
                    np.float_(tiempo.title()),
                    np.float_(temperatura.title()),
                    np.float_(altura.title())]
        predictS = model_prediction(x_in, model)
        st.success('La predicción del esfuerzo realizado es: {}'.format(predictS[0]).upper())

if __name__ == '__main__':
    main()
