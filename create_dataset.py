import pandas as pd
import mediapipe as mp
import cv2
import os

def distancia(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

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

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

dataset = []
base_path = "D:/Upao/Tesis/Dataset/somatotipos_dataset"  # Ajusta esta ruta a tu carpeta

for tipo in os.listdir(base_path):
    carpeta = os.path.join(base_path, tipo)
    for archivo in os.listdir(carpeta):
        if archivo.endswith((".png", ".jpg", ".jpeg")):
            ruta = os.path.join(carpeta, archivo)
            imagen = cv2.imread(ruta)
            if imagen is None:
                continue
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            resultado = pose.process(imagen_rgb)
            if resultado.pose_landmarks:
                proporciones = calcular_proporciones(resultado.pose_landmarks.landmark)
                proporciones["label"] = tipo  # Cambiado aquí
                dataset.append(proporciones)

df = pd.DataFrame(dataset)
df.to_csv("somatotipos_dataset_completo.csv", index=False)
print("Dataset guardado como somatotipos_dataset_completo.csv")
