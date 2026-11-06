# Aprendizaje Automático 3 · Examen P2 · Equipo N1

**Equipo:** León Franco Fernando, Mendoza Rodriguez Rodrigo, Muñoz Delgado Gael Arturo

Este repositorio contiene dos proyectos de clasificación. Ambos se desarrollaron en Google Colab con GPU T4 y se evaluaron en la prueba del cliente con umbral de 0.5.

| Proyecto | Cliente | Tarea | Modelo elegido | Resultado en prueba |
|---|---|---|---|---|
| [`ecg/`](ecg/) | CardioRitmo | Detectar latidos anormales | Conv1D | Cumple |
| [`resenas/`](resenas/) | Compra Fácil Bajío | Detectar reseñas negativas | BETO ajustado | Cumple |

## Video

**[Ver el video: dos clientes, un método (1:43)](video/examen_p2_dos_clientes.mp4)**

Cuenta los dos proyectos como una propuesta para cada cliente: el problema, los modelos que comparamos, la elección con validación, la prueba abierta una sola vez y los límites de cada modelo. Las cifras salen de los cuadernos y de los resultados de este repositorio. No tiene narración: el texto va en pantalla.

## Modelo de reseñas

El modelo BETO pesa 420 MB y supera el límite de 100 MB por archivo de GitHub, por lo que `model.safetensors` se publica aparte:

**[Descargar la carpeta del modelo desde Google Drive](https://drive.google.com/drive/folders/1kN9eWLKd9fQhWvdKGPZ7JwqujOL6fKuw?usp=sharing)**

Para usarlo, coloca los archivos descargados en `resenas/modelos_resena_beto/`. Esa carpeta debe quedar con `config.json`, `model.safetensors`, `tokenizer.json` y `tokenizer_config.json`. Después se carga con `from_pretrained`:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ruta = "resenas/modelos_resena_beto"
tokenizador = AutoTokenizer.from_pretrained(ruta)
modelo = AutoModelForSequenceClassification.from_pretrained(ruta)
```

En este modelo, la clase 1 significa reseña **negativa**.

## CardioRitmo: tamizaje de latidos (`ecg/`)

**Datos:** ECG5000 del archivo UCR, con 4,998 latidos de 140 puntos de un solo paciente (BIDMC, PhysioNet).

**Especificación del cliente:** sensibilidad y especificidad de al menos 95 %, con un máximo de 20,000 parámetros.

| Modelo | Tipo | Parámetros | Sensibilidad | Especificidad | AUC | Cumple |
|---|---|---:|---:|---:|---:|---|
| Línea base (todo normal) | línea base | 0 | 0.0 % | 100.0 % | 0.5000 | No |
| **Conv1D (elegido)** | convolucional | 5,857 | 99.3 % | 99.5 % | 0.9986 | Sí |
| TCN | convolucional | 14,593 | 98.8 % | 99.5 % | 0.9990 | Sí |
| Bi-GRU | recurrente | 6,785 | 99.0 % | 99.8 % | 0.9988 | Sí |

**Limitación principal:** todos los latidos vienen de una sola persona, así que no hay evidencia de que el modelo funcione con otros pacientes. Es una herramienta de apoyo para priorizar la revisión, no un diagnóstico.

## Compra Fácil Bajío: reseñas negativas (`resenas/`)

**Datos:** [SetFit/amazon_reviews_multi_es](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_es), reseñas de Amazon en español (Keung et al., 2020).

**Especificación del cliente:** exhaustividad de al menos 90 % y precisión de al menos 85 %, con un modelo propio, sin LLM por API.

| Modelo | Tipo | Parámetros | Exhaustividad | Precisión | Exactitud | Cumple |
|---|---|---:|---:|---:|---:|---|
| Línea base (todo negativa) | línea base | 0 | 100.0 % | 50.0 % | 50.0 % | No |
| TF-IDF + red densa | conjunto | 960,065 | 88.4 % | 92.8 % | 90.8 % | No |
| Embedding + Bi-LSTM | recurrente | 1,304,897 | 88.4 % | 89.3 % | 88.9 % | No |
| **BETO ajustado (elegido)** | transformer | 109,852,418 | 95.3 % | 92.5 % | 93.8 % | Sí |

**Limitaciones principales:** los datos son de Amazon España y el cliente es mexicano. El modelo no detectó ninguna de las 4 frases sarcásticas probadas. Además, la precisión baja cuando las reseñas negativas son menos frecuentes que en la prueba.

## Estructura del repositorio

```text
ecg/
├── DEV_ecg.ipynb            Cuaderno de desarrollo, con salidas
├── folleto_ecg.pdf          Folleto para el cliente
├── modelo_ecg.keras         Modelo elegido
├── resultados_ecg.csv       Resultados en la prueba del cliente
└── entrega_ecg.zip          Paquete de entrega
resenas/
├── DEV_resenas.ipynb        Cuaderno de desarrollo, con salidas
├── folleto_resenas.pdf      Folleto para el cliente
├── modelos_resena_beto/     Configuración y tokenizador (el modelo está en Drive)
├── resultados_resenas.csv   Resultados en la prueba del cliente
└── entrega_resenas.zip      Paquete de entrega
video/
└── examen_p2_dos_clientes.mp4  Video que explica los dos proyectos
```

## Reproducir los resultados

Abre `DEV_ecg.ipynb` o `DEV_resenas.ipynb` en Google Colab, activa la GPU T4 y ejecuta las celdas en orden. Los datos se descargan desde los propios cuadernos.
