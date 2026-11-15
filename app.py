
import streamlit as st
import base64
import streamlit.components.v1 as components
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2

from pathlib import Path

from PIL import Image, ImageOps, ImageFilter
from io import BytesIO

# ==========================================
# 1. CONFIGURACION
# ==========================================

st.set_page_config(
    page_title="Procesamiento de imagenes MNIST",
    page_icon="🔢",
    layout="wide"
)
# ==========================================
# CARGAR MODELO CNN
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

RUTA_MODELO = BASE_DIR / "modelos" / "cnn_mnist.keras"


@st.cache_resource
def cargar_modelo():

    modelo = tf.keras.models.load_model(
        str(RUTA_MODELO)
    )

    return modelo


if not RUTA_MODELO.exists():

    st.error("No se encontro el modelo CNN.")

    st.stop()


modelo = cargar_modelo()

st.title("Reconocimiento de digitos MNIST")

st.write(
    "Prueba de carga y procesamiento de imagenes "
    "para nuestra red neuronal convolucional."
)

st.divider()


# ==========================================
# 2. FUNCION PARA MOSTRAR IMAGENES
# ==========================================

def mostrar_imagen(imagen, caption, ancho=280):

    buffer = BytesIO()

    imagen.save(
        buffer,
        format="PNG"
    )

    imagen_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    html = f"""
    <html>
    <body style="
        margin: 0;
        padding: 10px;
        text-align: center;
        background: white;
    ">

        <img
            src="data:image/png;base64,{imagen_base64}"
            style="
                width: {ancho}px;
                max-width: 100%;
                height: auto;
                image-rendering: pixelated;
                object-fit: contain;
            "
        >

        <p style="
            color: #555;
            font-size: 14px;
        ">
            {caption}
        </p>

    </body>
    </html>
    """

    components.html(
        html,
        height=ancho + 65,
        scrolling=False
    )




def procesar_imagen(imagen):

    # ======================================
    # 1. CONVERTIR A RGB
    # ======================================

    imagen = ImageOps.exif_transpose(
        imagen
    ).convert("RGB")

    matriz_rgb = np.array(imagen)

    # ======================================
    # 2. CONVERTIR A LAB
    # ======================================

    matriz_lab = cv2.cvtColor(
        matriz_rgb,
        cv2.COLOR_RGB2LAB
    )

    alto, ancho = matriz_lab.shape[:2]

    # ======================================
    # 3. DETECTAR COLOR DEL FONDO
    # ======================================

    margen = max(1, int(min(alto, ancho) * 0.05))

    bordes = np.concatenate([
        matriz_lab[:margen, :, :].reshape(-1, 3),
        matriz_lab[-margen:, :, :].reshape(-1, 3),
        matriz_lab[:, :margen, :].reshape(-1, 3),
        matriz_lab[:, -margen:, :].reshape(-1, 3)
    ])

    color_fondo = np.median(
        bordes,
        axis=0
    )

    # ======================================
    # 4. CALCULAR DIFERENCIA DE COLORES
    # ======================================

    diferencia = np.sqrt(
        np.sum(
            (matriz_lab.astype(np.float32) - color_fondo) ** 2,
            axis=2
        )
    )

    diferencia = np.clip(
        diferencia,
        0,
        255
    ).astype(np.uint8)

    # ======================================
    # 5. SEPARAR NUMERO Y FONDO
    # ======================================

    _, mascara = cv2.threshold(
        diferencia,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ======================================
    # 6. ELIMINAR RUIDO
    # ======================================

    cantidad, etiquetas, estadisticas, centroides = (
        cv2.connectedComponentsWithStats(
            mascara,
            connectivity=8
        )
    )

    if cantidad <= 1:
        return None

    areas = estadisticas[1:, cv2.CC_STAT_AREA]

    area_mayor = areas.max()

    mascara_limpia = np.zeros_like(mascara)

    for i in range(1, cantidad):

        area = estadisticas[i, cv2.CC_STAT_AREA]

        if area >= max(8, area_mayor * 0.05):

            mascara_limpia[
                etiquetas == i
            ] = 255

    # ======================================
    # 7. DETECTAR LIMITES DEL NUMERO
    # ======================================

    filas, columnas = np.where(
        mascara_limpia > 0
    )

    if len(filas) == 0:
        return None

    x_min = columnas.min()
    x_max = columnas.max()

    y_min = filas.min()
    y_max = filas.max()

    recorte = Image.fromarray(
        mascara_limpia
    ).crop((
        x_min,
        y_min,
        x_max + 1,
        y_max + 1
    ))

    # ======================================
    # 8. REDIMENSIONAR A 20x20
    # ======================================

    recorte.thumbnail(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    # ======================================
    # 9. CREAR LIENZO DE 28x28
    # ======================================

    lienzo = Image.new(
        "L",
        (28, 28),
        color=0
    )

    x = (28 - recorte.width) // 2
    y = (28 - recorte.height) // 2

    lienzo.paste(
        recorte,
        (x, y)
    )

    return lienzo


# ==========================================
# 4. CARGAR IMAGEN
# ==========================================

st.subheader("1. Cargar imagen")

archivo = st.file_uploader(
    "Selecciona una imagen de un digito manuscrito",
    type=["png", "jpg", "jpeg"]
)



if archivo is not None:

    try:

        # ======================================
        # 1. CARGAR IMAGEN ORIGINAL
        # ======================================

        imagen_original = ImageOps.exif_transpose(
            Image.open(archivo)
        ).convert("RGB")

        # ======================================
        # 2. PROCESAR IMAGEN
        # ======================================

        imagen_procesada = procesar_imagen(
            imagen_original
        )

        if imagen_procesada is None:

            st.warning(
                "No se detecto ningun digito."
            )

        else:

            # ======================================
            # 3. MOSTRAR IMAGENES
            # ======================================

            st.divider()

            st.subheader("2. Comparacion de imagenes")

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("**Imagen original**")

                st.write(
                    f"Dimensiones: {imagen_original.width} "
                    f"x {imagen_original.height} pixeles"
                )

                mostrar_imagen(
                    imagen_original,
                    "Fotografia original",
                    ancho=300
                )

            with col2:

                st.markdown("**Imagen procesada**")

                st.write("Dimensiones: 28 x 28 pixeles")

                # Ampliar solo para visualizar
                imagen_ampliada = imagen_procesada.resize(
                    (280, 280),
                    Image.Resampling.NEAREST
                )

                mostrar_imagen(
                    imagen_ampliada,
                    "Imagen procesada a 28 x 28",
                    ancho=280
                )

            # ======================================
            # 4. PREPARAR IMAGEN PARA LA CNN
            # ======================================

            entrada = np.array(
                imagen_procesada
            ).astype("float32") / 255.0

            entrada = entrada.reshape(
                1, 28, 28, 1
            )

            # ======================================
            # 5. REALIZAR PREDICCION
            # ======================================

            probabilidades = modelo.predict(
                entrada,
                verbose=0
            )[0]

            prediccion = int(
                np.argmax(probabilidades)
            )

            confianza = float(
                probabilidades[prediccion] * 100
            )

            # ======================================
            # 6. MOSTRAR RESULTADO
            # ======================================

            st.divider()

            st.subheader("3. Resultado del reconocimiento")

            col1, col2 = st.columns(2)

            with col1:

                st.success(
                    f"Digito reconocido: {prediccion}"
                )

                st.title(str(prediccion))

            with col2:

                st.metric(
                    "Probabilidad estimada",
                    f"{confianza:.2f}%"
                )

            # ======================================
            # 7. PROBABILIDADES POR DIGITO
            # ======================================

            
            # ==========================================
            # 7. PROBABILIDADES POR DIGITO
            # ==========================================

            st.subheader("4. Probabilidades por digito")

            datos = pd.DataFrame({
                "Digito": [str(i) for i in range(10)],
                "Probabilidad": probabilidades * 100
            })

            # Mostrar grafica
            st.bar_chart(
                datos.set_index("Digito")
            )

            # ==========================================
            # TABLA DE PORCENTAJES
            # ==========================================

            st.subheader("Detalle de probabilidades")

            # Ordenar de mayor a menor
            datos_tabla = datos.sort_values(
                by="Probabilidad",
                ascending=False
            ).copy()

            # Formatear porcentajes
            datos_tabla["Probabilidad"] = datos_tabla[
                "Probabilidad"
            ].apply(lambda valor: f"{valor:.2f}%")

            # Mostrar tabla
            st.dataframe(
                datos_tabla,
                use_container_width=True,
                hide_index=True
            )


            # ======================================
            # 8. INFORMACION DEL PROCESAMIENTO
            # ======================================

            st.divider()

            st.subheader("5. Informacion del procesamiento")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric("Ancho", "28 px")

            with col2:

                st.metric("Alto", "28 px")

            with col3:

                st.metric("Canales", "1 (Grises)")

            st.success(
                "Prediccion realizada correctamente."
            )

    except Exception as error:

        st.error(
            f"Error al procesar la imagen: {error}"
        )
