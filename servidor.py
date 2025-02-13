from flask import Flask, render_template, request, redirect, url_for, Response



servidor = Flask(_name_)

@servidor.route("/")
def index():
    return render_template("index.html")


@servidor.route("/bienvenida")
def bienvenida():

    nombre = request.args.get('nombre', 'Invitado')

    # if(nombre == "Osvalo"):



    return render_template("bienvenida.html", nombre=nombre)


if _name_ == "_main_":
    servidor.run(port=4000, debug=True)