import sys
import json
import joblib
import numpy as np
import os

def load_model_and_scaler(model_path, scaler_path):
    try:
        kmeans = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return kmeans, scaler
    except Exception as e:
        print(json.dumps({'error': str(e)}), file=sys.stderr)
        sys.exit(1)

def scale_user_data(scaler, user_data):
    try:
        scaled_data = scaler.transform([[
            user_data['total_gastado'],
            user_data['total_comprado'],
            user_data['num_pedidos'],
            user_data['dias_entre_pedidos'],
            user_data['dias_activo']
        ]])
        return scaled_data
    except Exception as e:
        print(json.dumps({'error': str(e)}), file=sys.stderr)
        sys.exit(1)

def predict_cluster(kmeans, scaled_data):
    try:
        cluster = kmeans.predict(scaled_data)
        return cluster
    except Exception as e:
        print(json.dumps({'error': str(e)}), file=sys.stderr)
        sys.exit(1)

def main():
    # Obtener la ruta absoluta del directorio actual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Rutas absolutas de los archivos del modelo y el scaler
    model_path = os.path.join(current_dir, 'kmeans_model.pkl')
    scaler_path = os.path.join(current_dir, 'scaler.pkl')
    
    # Cargar el modelo y el scaler
    kmeans, scaler = load_model_and_scaler(model_path, scaler_path)
    
    # Leer datos del usuario desde stdin
    user_data = json.loads(sys.stdin.read())
    
    # Escalar los datos del usuario
    scaled_data = scale_user_data(scaler, user_data)
    
    # Predecir el cluster
    cluster = predict_cluster(kmeans, scaled_data)
    
    # Devolver el resultado
    print(json.dumps({'cluster': int(cluster[0])}))

if __name__ == "__main__":
    main()
