"""Crea datos/latidos_ejemplo.csv con latidos de la prueba del cliente, solo para demostrar la app.

Repite la carga y la división del examen. Estos latidos no se usan para decidir nada del modelo.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

URL = "https://storage.googleapis.com/download.tensorflow.org/data/ecg.csv"
SALIDA = Path(__file__).resolve().parent / "datos" / "latidos_ejemplo.csv"
CANTIDAD = 200

ecg = pd.read_csv(URL, header=None)
X = ecg.iloc[:, :-1].to_numpy("float32")
y = 1 - ecg.iloc[:, -1].to_numpy("int64")  # El archivo trae 1 = normal; aquí 1 = ANORMAL.
_, X_prueba, _, y_prueba = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

tabla = pd.DataFrame(X_prueba, columns=[f"p{i}" for i in range(1, 141)]).round(5)
tabla["etiqueta"] = y_prueba
muestra = tabla.sample(CANTIDAD, random_state=42).reset_index(drop=True)
SALIDA.parent.mkdir(exist_ok=True)
muestra.to_csv(SALIDA, index=False)
print(f"{SALIDA}: {len(muestra)} latidos, {muestra['etiqueta'].mean():.1%} anormales")
