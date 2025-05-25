from flask import Flask, request, jsonify
import mediapipe as mp
import cv2
import numpy as np
import base64
import joblib
import tempfile
import pandas as pd
from datetime import datetime
from supabase import create_client, Client
import os

# Configura tus credenciales de Supabase
SUPABASE_URL = "https://uezdldbsxoqzwpbslhob.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVlemRsZGJzeG9xendwYnNsaG9iIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0ODIwMzI1NywiZXhwIjoyMDYzNzc5MjU3fQ.vD-ByIgei8Gnw_FSptAAw-zFFy7anDLA3YsvL2k_g1Y"  # Sustituye por tu clave service_role si es backend
SUPABASE_BUCKET = "imagenes"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Inicializa Flask y modelos
app = Flask(__name__)
modelo = joblib.load("modelo_somatotipos_v2.pkl")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def distancia(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

def calcular_proporciones(landmarks):
    altura = distancia(landmarks[0], landmarks[32])
    torso = distancia(landmarks[0], landmarks[24])
    piernas = distancia(landmarks[24], landmarks[32])
    hombros = distancia(landmarks[11], landmarks[12])
    cintura = distancia(landmarks[23], landmarks[24])
    brazo_izq = distancia(landmarks[11], landmarks[13]) + distancia(landmarks[13], landmarks[15])
    brazo_der = distancia(landmarks[12], landmarks[14]) + distancia(landmarks[14], landmarks[16])
    brazo_prom = (brazo_izq + brazo_der) / 2
    pierna_izq = distancia(landmarks[23], landmarks[25]) + distancia(landmarks[25], landmarks[27])
    pierna_der = distancia(landmarks[24], landmarks[26]) + distancia(landmarks[26], landmarks[28])
    pierna_prom = (pierna_izq + pierna_der) / 2

    return {
        "altura": altura,
        "torso": torso,
        "piernas": piernas,
        "hombros": hombros,
        "cintura": cintura,
        "brazo_prom": brazo_prom,
        "pierna_prom": pierna_prom,
        "shoulder_waist_ratio": hombros / cintura if cintura != 0 else 0,
        "leg_height_ratio": piernas / altura if altura != 0 else 0,
        "torso_height_ratio": torso / altura if altura != 0 else 0,
        "brazo_altura_ratio": brazo_prom / altura if altura != 0 else 0,
        "pierna_altura_ratio": pierna_prom / altura if altura != 0 else 0,
        "hombro_torso_ratio": hombros / torso if torso != 0 else 0
    }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data["image"]
        sexo = data.get("sex", "Desconocido")

        # Decodificar imagen
        decoded_data = base64.b64decode(image_data)
        np_arr = np.frombuffer(decoded_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Procesar imagen con MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)

        if not result.pose_landmarks:
            return jsonify({"error": "No se detectaron landmarks"}), 400

        # Calcular proporciones
        proporciones = calcular_proporciones(result.pose_landmarks.landmark)
        df = pd.DataFrame([proporciones])
        pred = modelo.predict(df)[0]
        proba = modelo.predict_proba(df).max()

        # Guardar imagen en Supabase (bucket privado)
        nombre_imagen = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
            cv2.imwrite(temp_img.name, img)
            with open(temp_img.name, "rb") as f:
                supabase.storage.from_(SUPABASE_BUCKET).upload(
                    path=nombre_imagen,
                    file=f,
                    file_options={"content-type": "image/jpeg", "x-upsert": "true"}  # cadena, no booleano
                )

        imagen_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{nombre_imagen}"

        # Insertar en la base de datos
        supabase.table("registros").insert({
            "sex": sexo,
            "label": pred,
            "precision": float(proba),
            "shoulder_waist_ratio": proporciones["shoulder_waist_ratio"],
            "leg_height_ratio": proporciones["leg_height_ratio"],
            "torso_height_ratio": proporciones["torso_height_ratio"],
            "brazo_altura_ratio": proporciones["brazo_altura_ratio"],
            "pierna_altura_ratio": proporciones["pierna_altura_ratio"],
            "hombro_torso_ratio": proporciones["hombro_torso_ratio"],
            "imagen": imagen_url
        }).execute()

        return jsonify({"somatotipo": pred})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
