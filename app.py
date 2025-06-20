from flask import Flask, render_template, request, redirect, url_for
from modelo import predecir_medicamento
from modelo import precision
import numpy as np

app = Flask(__name__)

preguntas = ["¿Cuál es tu edad?", "¿Cuál es tu sexo? (M(1)/F(0))", "¿Cuál es tu presión arterial? (HIGH(1)/LOW(0))","¿Cuál es tu nivel de colesterol? (HIGH(0)/LOW(1))","¿Cuál es tu nivel en potasio?"]
respuestas = []




@app.route('/')
def index():
    return render_template('sitio/index.html', pregunta=preguntas[0])

@app.route('/respuesta', methods=['POST'])
def respuesta():
    respuesta = request.form['respuesta']
    respuestas.append(respuesta)

    if len(respuestas) < len(preguntas):
        return render_template('sitio/index.html', pregunta=preguntas[len(respuestas)])
    else:
        return redirect(url_for('fin'))

@app.route('/imprim', methods=['POST'])  
def imprim():
   imprim = precision()
       

@app.route('/fin')
def fin():
    # Procesar las respuestas
    # Aquí necesitas convertir las respuestas del usuario a un formato adecuado
    # para enviarlo a la función predecir_medicamento en modelo.py
    #datos_paciente = convertir_respuestas_a_numpy(respuestas)
    
    # Hacer la predicción usando la función del modelo
    resultado_prediccion = predecir_medicamento([respuestas])

    return render_template('sitio/fin.html', respuestas=resultado_prediccion)


if __name__ == '__main__':
    app.run(debug=True)

    