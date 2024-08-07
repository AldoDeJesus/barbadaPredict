import sys
import json
import joblib
import numpy as np
import os
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        try:
            # Obtener la ruta absoluta del directorio actual
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # Cargar el modelo y el scaler usando rutas absolutas
            kmeans = joblib.load(os.path.join(current_dir, 'kmeans_model.pkl'))
            scaler = joblib.load(os.path.join(current_dir, 'scaler.pkl'))
            
            # Leer datos del usuario desde la solicitud POST
            user_data = request.get_json()
            if user_data is None:
                return jsonify({"error": "No JSON data provided"}), 400
            
            # Escalar los datos del usuario
            scaled_data = scaler.transform([[
                user_data['total_gastado'],
                user_data['total_comprado'],
                user_data['num_pedidos'],
                user_data['dias_entre_pedidos'],
                user_data['dias_activo']
            ]])
            
            # Predecir el cluster
            cluster = kmeans.predict(scaled_data)
            
            # Devolver el resultado
            return jsonify({'cluster': int(cluster[0])})
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    else:
        # Manejo de solicitudes GET
        return jsonify({"message": "This endpoint expects POST requests with JSON data"}), 200

if __name__ == "__main__":
    app.run(debug=True)

