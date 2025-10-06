from flask import Flask, request, jsonify
from flask_cors import CORS
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
import time

# --------- Config ----------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE")  # SOLO backend
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "imagenes")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Faltan SUPABASE_URL o SUPABASE_SERVICE_ROLE en variables de entorno.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --------- App ----------
app = Flask(__name__)
CORS(app)  # si tienes front aparte; si no, puedes quitarlo

# Carga del modelo (ajusta el nombre si cambia)
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

@app.get("/")
def root():
    return {"ok": True}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post('/predict')
def predict():
    try:
        t0 = time.time()

        data = request.get_json(force=True)
        image_data = data["image"]
        sexo = data.get("sex", "Desconocido")

        decoded = base64.b64decode(image_data)
        np_arr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)

        if not result.pose_landmarks:
            return jsonify({"error": "No se detectaron landmarks"}), 400

        props = calcular_proporciones(result.pose_landmarks.landmark)
        df = pd.DataFrame([props])

        pred = modelo.predict(df)[0]
        proba = float(np.max(modelo.predict_proba(df)))

        # --- subir imagen a Storage ---
        nombre_imagen = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S%f')}.jpg"
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, img)
            with open(tmp.name, "rb") as f:
                supabase.storage.from_(SUPABASE_BUCKET).upload(
                    path=nombre_imagen,
                    file=f,
                    file_options={"content-type": "image/jpeg", "x-upsert": "true"}
                )

        # Si el bucket es público, la URL directa sirve:
        imagen_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{nombre_imagen}"

        # --- guardar registro ---
        supabase.table("registros").insert({
            "sex": sexo,
            "label": pred,
            "precision": proba,
            "shoulder_waist_ratio": props["shoulder_waist_ratio"],
            "leg_height_ratio": props["leg_height_ratio"],
            "torso_height_ratio": props["torso_height_ratio"],
            "brazo_altura_ratio": props["brazo_altura_ratio"],
            "pierna_altura_ratio": props["pierna_altura_ratio"],
            "hombro_torso_ratio": props["hombro_torso_ratio"],
            "imagen": imagen_url,
            "tiempo_procesamiento": round(time.time() - t0, 4)
        }).execute()

        return jsonify({"somatotipo": pred, "precision": proba, "imagen": imagen_url})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Para desarrollo local (en Koyeb ejecuta gunicorn desde Dockerfile)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
