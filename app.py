#from flask import Flask, request, render_template, jsonify/
#import joblib
#import pandas as pd
#import logging

#app = Flask(__name__)

# Configurar el registro
#logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
#model = joblib.load('modelo_random_forest.pkl')
#scaler = joblib.load('scaler.pkl')
#encoder = joblib.load('modelo_ordinalEncoder.pkl')
#app.logger.debug('Modelo cargado correctamente.')

#@app.route('/')
#def home():
#    return render_template('formulario.html')

#@app.route('/predict', methods=['POST'])
#def predict():
#    try:
        # Obtener los datos enviados en el request
 #       Pclass = float(request.form['Pclass'])
  #      Sex = float(request.form['Sex'])
   #     Age = float(request.form['Age'])
    #    SibSp = float(request.form['SibSp'])
     #   Embarked = float(request.form['Embarked'])



        # Crear un DataFrame con los datos
      #  data_df = pd.DataFrame([[Pclass, Sex, Age, SibSp, Embarked]], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Embarked'])
       # app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        #prediction = model.predict(data_df)
        #app.logger.debug(f'Predicción: {prediction[0]}')
        
        # Devolver las predicciones como respuesta JSON
        #return jsonify({'categoria': prediction[0]})
    #except Exception as e:
     #   app.logger.error(f'Error en la predicción: {str(e)}')
      #  return jsonify({'error': str(e)}), 400

#if __name__ == '__main__':
 #   app.run(debug=True)


from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado, el escalador y el codificador
model = joblib.load('modelo_random_forest.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('modelo_ordinalEncoder.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        Pclass = float(request.form['Pclass'])
        Sex = request.form['Sex']
        Age = float(request.form['Age'])
        SibSp = float(request.form['SibSp'])
        Embarked = request.form['Embarked']

        # Crear un DataFrame con los datos
        data_dict = {'Pclass': Pclass, 'Sex': Sex, 'Age': Age, 'SibSp': SibSp, 'Embarked': Embarked}
        data_df = pd.DataFrame([data_dict])

        # Aplicar el codificador ordinal
        data_df[['Sex', 'Embarked']] = encoder.transform(data_df[['Sex', 'Embarked']])

        # Aplicar el escalador
        data_df_scaled = scaler.transform(data_df)

        # Realizar predicciones
        prediction = model.predict(data_df_scaled)
        app.logger.debug(f'Predicción: {prediction[0]}')

        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoria': int(prediction[0])})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
