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

app = Flask(__name__)

# Inicializa Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_BUCKET = "imagenes"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Carga diferida del modelo
modelo = None

def cargar_modelo():
    global modelo
    if modelo is None:
        modelo = joblib.load("modelo_somatotipos_v2.pkl")
        print("Modelo cargado correctamente.")

# Resto de funciones como distancia(), calcular_proporciones(), etc.

@app.route('/predict', methods=['POST'])
def predict():
    try:
        cargar_modelo()  # << Cargar solo si no está cargado

        data = request.get_json()
        image_data = data["image"]
        sexo = data.get("sex", "Desconocido")

        decoded_data = base64.b64decode(image_data)
        np_arr = np.frombuffer(decoded_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)

        if not result.pose_landmarks:
            return jsonify({"error": "No se detectaron landmarks"}), 400

        proporciones = calcular_proporciones(result.pose_landmarks.landmark)
        df = pd.DataFrame([proporciones])
        pred = modelo.predict(df)[0]
        proba = modelo.predict_proba(df).max()

        nombre_imagen = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
            cv2.imwrite(temp_img.name, img)
            with open(temp_img.name, "rb") as f:
                supabase.storage.from_(SUPABASE_BUCKET).upload(
                    path=nombre_imagen,
                    file=f,
                    file_options={"content-type": "image/jpeg", "x-upsert": "true"}
                )

        imagen_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{nombre_imagen}"

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
