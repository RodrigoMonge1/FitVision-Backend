import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Cargar el dataset actualizado
df = pd.read_csv("somatotipos_dataset_completo.csv")

# Separar características y etiquetas
X = df.drop(columns=["label"])
y = df["label"]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Entrenar modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Evaluar modelo
y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))

# Guardar modelo
joblib.dump(modelo, "modelo_somatotipos_v3.pkl")
print("Modelo entrenado y guardado como modelo_somatotipos_v3.pkl")
