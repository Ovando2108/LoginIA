import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Configuración de parámetros
ancho_imagen, alto_imagen = 128, 128  
tamano_lote = 60  
epocas = 100

directorio_entrenamiento = "C:/Users/osval/Documents/IA/LoginIA/dataset/entrenamiento"
directorio_validacion = "C:/Users/osval/Documents/IA/LoginIA/dataset/validacion"

# Generadores de imágenes con aumento de datos
generador_entrenamiento = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=50,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=15,
    zoom_range=[0.5, 1.5],
    vertical_flip=True,
    horizontal_flip=True,
)

generador_validacion = ImageDataGenerator(rescale=1.0 / 255)

# Carga de datos desde las carpetas
datos_entrenamiento = generador_entrenamiento.flow_from_directory(
    directorio_entrenamiento,
    target_size=(ancho_imagen, alto_imagen),
    batch_size=tamano_lote,
    class_mode="categorical",
)

datos_validacion = generador_validacion.flow_from_directory(
    directorio_validacion,
    target_size=(ancho_imagen, alto_imagen),
    batch_size=tamano_lote,
    class_mode="categorical",
)

# Definición de la red neuronal convolucional
modelo = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(ancho_imagen, alto_imagen, 3)), #capa de convolucion
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"), #capa de reducción
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.5), # desactiva la mitad de las neuronas p
    layers.Flatten(),
    layers.Dense(128, activation="relu"), #capa completamente conectada 
    layers.Dense(len(datos_entrenamiento.class_indices), activation="softmax")
])

# Compilación del modelo
modelo.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# entrenamiento y validacion de datos 
historial = modelo.fit(
    datos_entrenamiento,
    epochs=epocas,
    validation_data=datos_validacion
)

# Evaluación del modelo
puntaje = modelo.evaluate(datos_validacion)
print(f"Pérdida: {puntaje[0]:.4f}, Precisión: {puntaje[1]:.4f}")

# Guardado del modelo y pesos
modelo.save("modelo_clasificador_personas.h5")
modelo.save_weights("pesos.weights.h5")