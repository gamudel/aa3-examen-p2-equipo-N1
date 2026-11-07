"""Carga de los modelos guardados en modelo/ e inferencia, sin depender de la interfaz."""
import io
import os
from pathlib import Path

# Keras 3 puede usar PyTorch como motor, así no hace falta instalar TensorFlow.
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RUTA_ECG = RAIZ / "ecg" / "modelo" / "modelo_ecg.keras"
RUTA_BETO = RAIZ / "resenas" / "modelo" / "modelo_resenas_beto"
RUTA_EJEMPLOS = Path(__file__).resolve().parent / "datos" / "latidos_ejemplo.csv"
ENLACE_DRIVE = "https://drive.google.com/drive/folders/1kN9eWLKd9fQhWvdKGPZ7JwqujOL6fKuw?usp=sharing"

UMBRAL = 0.5  # El mismo umbral con el que se midió la prueba del cliente.
LARGO_LATIDO = 140
MAX_LARGO_BETO = 128  # El mismo largo con el que se entrenó BETO.


def cargar_ecg(ruta=RUTA_ECG):
    import keras

    # Solo se usa para predecir, no hace falta el optimizador con el que se entrenó.
    return keras.models.load_model(ruta, compile=False)


def probabilidad_latidos(modelo, X, tam_lote=256):
    """Probabilidad de que cada latido sea anormal. X tiene forma (n, 140)."""
    import keras

    # Se llama al modelo directo: predict() con el motor de PyTorch intenta compilar y pide un
    # compilador de C++, que en Windows no suele estar instalado.
    X = np.asarray(X, dtype="float32")[..., None]
    lotes = [keras.ops.convert_to_numpy(modelo(X[inicio:inicio + tam_lote], training=False))
             for inicio in range(0, len(X), tam_lote)]
    return np.concatenate(lotes).ravel()


def leer_latidos(archivo):
    """Lee un CSV de 140 columnas, con una columna extra opcional de etiqueta (1 = anormal).

    Devuelve (X, etiquetas), donde etiquetas puede ser None. Lanza ValueError si el formato no sirve.
    """
    try:
        tabla = pd.read_csv(archivo, header=None)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError("No se pudo leer el archivo como CSV.") from error

    if not tabla.empty and pd.to_numeric(tabla.iloc[0], errors="coerce").isna().all():
        tabla = tabla.iloc[1:]  # La primera fila es un encabezado.
    if tabla.empty:
        raise ValueError("El archivo no tiene latidos.")
    if tabla.shape[1] not in (LARGO_LATIDO, LARGO_LATIDO + 1):
        raise ValueError(
            f"Cada fila debe tener {LARGO_LATIDO} puntos, más una etiqueta opcional. "
            f"El archivo tiene {tabla.shape[1]} columnas."
        )

    numeros = tabla.apply(pd.to_numeric, errors="coerce")
    if numeros.isna().any().any():
        raise ValueError("Hay celdas vacías o que no son números.")

    X = numeros.iloc[:, :LARGO_LATIDO].to_numpy("float32")
    if not np.isfinite(X).all():
        raise ValueError("Hay valores infinitos en los latidos.")

    etiquetas = None
    if tabla.shape[1] == LARGO_LATIDO + 1:
        etiquetas = numeros.iloc[:, LARGO_LATIDO].to_numpy()
        if not np.isin(etiquetas, [0, 1]).all():
            raise ValueError("La última columna debe ser una etiqueta: 0 = normal, 1 = anormal.")
        etiquetas = etiquetas.astype(int)
    return X, etiquetas


def parece_normalizado(X, tolerancia=0.25):
    """El modelo se entrenó con latidos de media 0 y desviación 1, cada uno por separado."""
    return (np.abs(X.mean(axis=1)).mean() < tolerancia
            and np.abs(X.std(axis=1) - 1).mean() < tolerancia)


def leer_resenas(archivo):
    """Lee un CSV con una columna de reseñas (texto, text, resena o reseña) o con una sola columna."""
    contenido = archivo.read()
    tabla = None
    for codificacion in ("utf-8-sig", "latin-1"):
        try:
            tabla = pd.read_csv(io.BytesIO(contenido), encoding=codificacion)
            break
        except UnicodeDecodeError:
            continue
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
            raise ValueError("No se pudo leer el archivo como CSV.") from error

    columnas = {str(columna).strip().lower(): columna for columna in tabla.columns}
    nombre = next((columnas[n] for n in ("texto", "text", "resena", "reseña") if n in columnas), None)
    if nombre is None:
        if tabla.shape[1] != 1:
            raise ValueError("El CSV debe tener una columna llamada texto, o una sola columna.")
        nombre = tabla.columns[0]
    textos = [t.strip() for t in tabla[nombre].dropna().astype(str) if t.strip()]
    if not textos:
        raise ValueError("El archivo no tiene reseñas.")
    return textos


def beto_disponible(ruta=RUTA_BETO):
    return (Path(ruta) / "model.safetensors").is_file()


def cargar_beto(ruta=RUTA_BETO):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizador = AutoTokenizer.from_pretrained(ruta)
    modelo = AutoModelForSequenceClassification.from_pretrained(ruta)
    modelo.eval()
    return tokenizador, modelo


def probabilidad_resenas(tokenizador, modelo, textos, tam_lote=16):
    """Probabilidad de que cada reseña sea negativa. En este modelo, la clase 1 es negativa."""
    import torch

    probabilidades = []
    with torch.no_grad():
        for inicio in range(0, len(textos), tam_lote):
            lote = [str(texto) for texto in textos[inicio:inicio + tam_lote]]
            entradas = tokenizador(lote, truncation=True, max_length=MAX_LARGO_BETO,
                                   padding=True, return_tensors="pt")
            logits = modelo(**entradas).logits
            probabilidades.append(torch.softmax(logits, dim=-1)[:, 1].numpy())
    return np.concatenate(probabilidades) if probabilidades else np.array([])
