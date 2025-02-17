from flask import Flask, request, render_template, redirect, url_for
import os
from prediccion import predecir_imagen

app = Flask(__name__)

UPLOAD_FOLDER = "imagenes_subidas"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

usuarios = {"persona1": "Persona A", "persona2": "Persona B", "persona3": "Persona C"}

@app.route("/")
def index():
    return render_template("index.html", usuarios=usuarios)

@app.route("/seleccionar/<usuario>")
def seleccionar(usuario):
    if usuario not in usuarios:
        return "Usuario no válido", 400
    return render_template("subir_imagen.html", usuario=usuario)

@app.route("/predecir", methods=["POST"])
def predecir():
    if "archivo" not in request.files or "usuario" not in request.form:
        return "Faltan datos", 400
    
    archivo = request.files["archivo"]
    usuario = request.form["usuario"]
    
    if usuario not in usuarios:
        return "Usuario no válido", 400
    
    if archivo.filename == "":
        return "Nombre de archivo vacío", 400
    
    ruta_imagen = os.path.join(app.config["UPLOAD_FOLDER"], archivo.filename)
    archivo.save(ruta_imagen)
    
    # PASAR USUARIO SELECCIONADO A LA FUNCIÓN
    mensaje, _ = predecir_imagen(ruta_imagen, usuario)
    
    return f"<h2>{mensaje}</h2>", 200

    persona_predicha, probabilidad = predecir_imagen(ruta_imagen)
    
    if usuarios[usuario] == persona_predicha:
        return f"<h2>Bienvenido {usuarios[usuario]}!</h2>", 200
    else:
        return "<h2>Acceso denegado. No eres la persona seleccionada.</h2>", 403

if __name__ == "__main__":
    app.run(debug=True)
