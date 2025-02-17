import tensorflow as tf
import numpy as np
import cv2

# Cargar el modelo entrenado
modelo = tf.keras.models.load_model("modelo_clasificador_personas.h5")

# Dimensiones de entrada del modelo
ancho_imagen, alto_imagen = 128, 128  

# Diccionario de clases
nombres_clases = {0: "persona1", 1: "persona2", 2: "persona3"}

# Umbral de confianza (ajústalo según pruebas)
UMBRAL = 0.70

def predecir_imagen(ruta_imagen, usuario_seleccionado):
    """
    Realiza la predicción de la imagen y verifica si coincide con la persona seleccionada.
    """
    # Cargar y preprocesar la imagen
    imagen = cv2.imread(ruta_imagen)
    imagen = cv2.resize(imagen, (ancho_imagen, alto_imagen))
    imagen = imagen.astype("float32") / 255.0  
    imagen = np.expand_dims(imagen, axis=0)  # Añadir dimensión del lote

    # Hacer predicción
    probabilidades = modelo.predict(imagen)[0]  # Obtener el vector de probabilidades
    clase_predicha = np.argmax(probabilidades)
    persona_predicha = nombres_clases[clase_predicha]
    probabilidad_predicha = probabilidades[clase_predicha]

    # Determinar si la predicción es confiable
    es_correcto = (persona_predicha == usuario_seleccionado) and (probabilidad_predicha >= UMBRAL)
    
    if es_correcto:
        mensaje = f"✅ Acceso permitido: Bienvenido, {persona_predicha} ({probabilidad_predicha*100:.2f}%)"
    else:
        mensaje = f"❌ Acceso denegado: No eres {usuario_seleccionado} (mejor coincidencia: {persona_predicha} con {probabilidad_predicha*100:.2f}%)"

    return mensaje, {nombres_clases[i]: f"{prob * 100:.2f}%" for i, prob in enumerate(probabilidades)}
