"""PMV de los dos clientes del examen. Se corre con: streamlit run pmv/app.py"""
import hashlib
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import modelos

AZUL = "#2a78d6"     # Sin marcar / referencia
NARANJA = "#eb6834"  # Marcado por el modelo
LIMITE_RESENAS = 2000
OPCIONES_LISTA = {"10": 10, "25": 25, "50": 50, "Todos": None}

EJEMPLOS_RESENAS = """El paquete llegó abierto y le faltaba una pieza. Ya escribí dos veces.
Pedí talla mediana, me mandaron otra y nadie me contesta.
Se ve bonito, pero a la semana dejó de prender y no me dan respuesta.
El producto está bien, pero tardó tres semanas en llegar.
Muy buena calidad, llegó antes de lo esperado y lo recomiendo.
Está bien chido, me encantó.
Jala al cien y llegó rapidísimo.
Qué maravilla, se rompió el primer día."""

st.set_page_config(page_title="PMV · Equipo N1", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Cargando el modelo de latidos...")
def modelo_ecg():
    return modelos.cargar_ecg(modelos.RUTA_ECG)


@st.cache_resource(show_spinner="Cargando BETO, tarda unos segundos la primera vez...")
def modelo_beto():
    return modelos.cargar_beto(modelos.RUTA_BETO)


def huella(*partes):
    return hashlib.sha1("\x00".join(partes).encode("utf-8")).hexdigest()[:10]


def acciones(prefijo, indice, boton_si, boton_no, texto_si, texto_no):
    """Dos botones cuya decisión vive solo en esta sesión. No se envía ni se guarda nada."""
    clave = f"{prefijo}:{indice}"
    izquierda, derecha = st.columns(2)
    if izquierda.button(boton_si, key=f"{clave}:si", width="stretch"):
        st.session_state[clave] = texto_si
    if derecha.button(boton_no, key=f"{clave}:no", width="stretch"):
        st.session_state[clave] = texto_no
    decision = st.session_state.get(clave)
    if decision:
        st.caption(f"Decisión: {decision}")


def grafica_probabilidades(prob):
    marcado = prob >= modelos.UMBRAL
    figura = go.Figure()
    for mascara, nombre, color, simbolo in [(~marcado, "Sin marcar", AZUL, "circle"),
                                            (marcado, "Marcado", NARANJA, "diamond")]:
        figura.add_scatter(
            x=np.flatnonzero(mascara) + 1, y=prob[mascara], mode="markers", name=nombre,
            marker=dict(color=color, size=8, symbol=simbolo),
            hovertemplate="Latido %{x}<br>Prob. anormal %{y:.3f}<extra>" + nombre + "</extra>",
        )
    figura.add_hline(y=modelos.UMBRAL, line_dash="dot", line_width=1, line_color="gray",
                     annotation_text="Umbral 0.5", annotation_position="top left")
    figura.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10), hovermode="closest",
                         legend=dict(orientation="h", y=1.18, x=1, xanchor="right"),
                         xaxis_title="Posición en el archivo", yaxis_title="Prob. anormal",
                         yaxis=dict(range=[-0.05, 1.05]))
    return figura


def grafica_latido(latido, referencia):
    figura = go.Figure()
    puntos = np.arange(1, len(latido) + 1)
    if referencia is not None:
        figura.add_scatter(x=puntos, y=referencia, mode="lines", name="Promedio sin marcar",
                           line=dict(color=AZUL, width=2, dash="dot"),
                           hovertemplate="%{y:.2f}<extra>Promedio sin marcar</extra>")
    figura.add_scatter(x=puntos, y=latido, mode="lines", name="Este latido",
                       line=dict(color=NARANJA, width=2),
                       hovertemplate="%{y:.2f}<extra>Este latido</extra>")
    figura.update_layout(height=170, margin=dict(l=5, r=5, t=25, b=5), hovermode="x unified",
                         legend=dict(orientation="h", y=1.25, x=1, xanchor="right", font=dict(size=11)),
                         xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
    return figura


def pestana_ecg():
    st.subheader("CardioRitmo · Revisión de latidos")
    st.info(
        "**Apoyo para priorizar la revisión, no un diagnóstico.** Un latido sin marcar no significa que "
        "esté sano, y la decisión clínica es del cardiólogo. Hoy solo hay evidencia con un paciente; "
        "no debe usarse con pacientes sin validación clínica.",
        icon="⚠️",
    )

    fuente = st.radio("Registro", ["Latidos de ejemplo", "Subir un CSV"], horizontal=True, key="ecg_fuente")
    if fuente == "Latidos de ejemplo":
        contenido = modelos.RUTA_EJEMPLOS.read_bytes()
        st.caption("200 latidos tomados al azar de la prueba del cliente (ECG5000). No son consecutivos "
                   "ni un registro real de 24 h.")
    else:
        archivo = st.file_uploader(
            "CSV con un latido por fila: 140 puntos ya normalizados y, si quieres, una última columna "
            "de etiqueta (1 = anormal)", type="csv", key="ecg_archivo")
        if archivo is None:
            return
        contenido = archivo.getvalue()

    try:
        X, etiquetas = modelos.leer_latidos(io.BytesIO(contenido))
    except ValueError as error:
        st.error(f"No se pudo usar el archivo. {error}")
        return
    if not modelos.parece_normalizado(X):
        st.warning("Estos latidos no parecen normalizados (media 0 y desviación 1 por latido). "
                   "El modelo se entrenó así, por lo que el resultado puede no ser confiable.")

    try:
        prob = modelos.probabilidad_latidos(modelo_ecg(), X)
    except (OSError, ValueError) as error:
        st.error(f"No se pudo cargar {modelos.RUTA_ECG.name}. {error}")
        return
    marcado = prob >= modelos.UMBRAL

    a, b, c = st.columns(3)
    a.metric("Latidos en el registro", f"{len(X):,}")
    b.metric("Marcados para revisar", f"{marcado.sum():,}")
    c.metric("Del registro", f"{marcado.mean():.1%}")

    st.plotly_chart(grafica_probabilidades(prob), width="stretch", key="ecg_dispersion")

    if etiquetas is not None:
        anormales, normales = etiquetas == 1, etiquetas == 0
        st.caption(
            f"Con las etiquetas del archivo: marcó {marcado[anormales].sum()} de {anormales.sum()} anormales "
            f"y {marcado[normales].sum()} de {normales.sum()} normales. Es una comprobación de este archivo; "
            "el desempeño medido está en el folleto."
        )

    st.markdown("#### Latidos marcados, de mayor a menor probabilidad")
    if not marcado.any():
        st.success("El modelo no marcó ningún latido en este registro. Eso no garantiza que todos estén sanos.")
        return

    cuantos = OPCIONES_LISTA[st.selectbox("Mostrar", list(OPCIONES_LISTA), key="ecg_cuantos")]
    orden = np.flatnonzero(marcado)[np.argsort(-prob[marcado])][:cuantos]
    referencia = X[~marcado].mean(axis=0) if (~marcado).any() else None
    prefijo = "ecg" + hashlib.sha1(contenido).hexdigest()[:10]
    for indice in orden:
        with st.container(border=True):
            izquierda, derecha = st.columns([3, 2], vertical_alignment="center")
            izquierda.plotly_chart(grafica_latido(X[indice], referencia), width="stretch",
                                   key=f"{prefijo}:g{indice}")
            with derecha:
                st.markdown(f"**Latido {indice + 1:,}**  \nProb. anormal **{prob[indice]:.2f}**")
                acciones(prefijo, int(indice), "Confirmar", "Descartar",
                         "confirmado por el revisor", "descartado por el revisor")


def pestana_resenas():
    st.subheader("Compra Fácil Bajío · Fila de atención")
    st.info(
        "**Sirve para ordenar la atención, no para decidir sobre las personas.** No debe borrar ni ocultar "
        "reseñas, ni responder solo. No detectó ninguna de 4 quejas sarcásticas probadas y aprendió de "
        "Amazon España, así que una reseña sin marcar también merece atención.",
        icon="⚠️",
    )

    if not modelos.beto_disponible(modelos.RUTA_BETO):
        st.error(
            f"Falta el modelo BETO (420 MB). [Descárgalo de Drive]({modelos.ENLACE_DRIVE}) y coloca "
            "`model.safetensors` en `resenas/modelo/modelo_resenas_beto/`. Después recarga esta página."
        )
        return

    if "resenas_texto" not in st.session_state:
        st.session_state.resenas_texto = ""

    def cargar_ejemplos():
        st.session_state.resenas_texto = EJEMPLOS_RESENAS

    st.text_area("Reseñas, una por línea", key="resenas_texto", height=180,
                 placeholder="Escribe o pega aquí las reseñas...")
    archivo = st.file_uploader("O sube un CSV con una columna llamada texto", type="csv", key="resenas_archivo")
    izquierda, derecha, _ = st.columns([1, 1, 3])
    clasificar = izquierda.button("Clasificar", type="primary", width="stretch")
    derecha.button("Usar ejemplos", on_click=cargar_ejemplos, width="stretch")
    st.caption("Las reseñas de ejemplo fueron escritas para esta demostración; no son de clientes reales.")

    if clasificar:
        textos = [linea.strip() for linea in st.session_state.resenas_texto.splitlines() if linea.strip()]
        if archivo is not None:
            try:
                textos += modelos.leer_resenas(io.BytesIO(archivo.getvalue()))
            except ValueError as error:
                st.error(f"No se pudo usar el archivo. {error}")
                return
        if not textos:
            st.warning("Escribe al menos una reseña o sube un archivo.")
            st.session_state.pop("resenas_resultado", None)
            return
        if len(textos) > LIMITE_RESENAS:
            st.warning(f"Este PMV corre en CPU: se clasifican solo las primeras {LIMITE_RESENAS:,} "
                       f"de {len(textos):,} reseñas.")
            textos = textos[:LIMITE_RESENAS]
        tokenizador, modelo = modelo_beto()
        with st.spinner(f"Clasificando {len(textos):,} reseñas..."):
            prob = modelos.probabilidad_resenas(tokenizador, modelo, textos)
        st.session_state.resenas_resultado = pd.DataFrame({"texto": textos, "prob": prob})

    resultado = st.session_state.get("resenas_resultado")
    if resultado is None:
        return

    marcadas = resultado[resultado["prob"] >= modelos.UMBRAL].sort_values("prob", ascending=False)
    resto = resultado[resultado["prob"] < modelos.UMBRAL].sort_values("prob", ascending=False)
    prefijo = "res" + huella(*resultado["texto"])

    a, b, c = st.columns(3)
    a.metric("Reseñas recibidas", f"{len(resultado):,}")
    b.metric("Marcadas como negativas", f"{len(marcadas):,}")
    metrica_pendientes = c.empty()

    st.markdown("#### Fila de reseñas marcadas, de mayor a menor probabilidad")
    if marcadas.empty:
        st.success("El modelo no marcó ninguna reseña.")
    cuantos = OPCIONES_LISTA[st.selectbox("Mostrar", list(OPCIONES_LISTA), key="resenas_cuantos")]
    for indice, fila in marcadas.head(cuantos if cuantos else len(marcadas)).iterrows():
        with st.container(border=True):
            izquierda, derecha = st.columns([3, 2], vertical_alignment="center")
            with izquierda:
                st.text(fila["texto"])
            with derecha:
                st.markdown(f"Prob. negativa **{fila['prob']:.2f}**")
                acciones(prefijo, indice, "Atendida", "No es queja",
                         "atendida por el equipo", "no es queja")

    # Se calcula al final para contar también el botón que se acaba de presionar.
    pendientes = sum(f"{prefijo}:{i}" not in st.session_state for i in marcadas.index)
    metrica_pendientes.metric("Por responder", f"{pendientes:,}")

    with st.expander(f"Reseñas sin marcar ({len(resto):,})"):
        st.dataframe(
            resto.rename(columns={"texto": "Reseña", "prob": "Prob. negativa"}),
            hide_index=True, width="stretch",
            column_config={"Prob. negativa": st.column_config.NumberColumn(format="%.2f")},
        )


st.title("Producto mínimo viable · Equipo N1")
st.caption(
    "Clasifica con los modelos guardados en `modelo/` y el umbral de 0.5 de la prueba del cliente. "
    "Empresas ficticias. Trabajo académico, IN1042 Aprendizaje Automático III."
)

latidos, resenas = st.tabs(["CardioRitmo: latidos", "Compra Fácil Bajío: reseñas"])
with latidos:
    pestana_ecg()
with resenas:
    pestana_resenas()
