from flask import Flask
from flask import request
from flask import render_template
import sys, os

import base64
from io import BytesIO
from matplotlib.figure import Figure

from model import graph

app = Flask(__name__)


# @app.route("/graph", methods=["GET", "POST"])
@app.route("/graph", methods=["GET", "POST"])
def draw_graph():
    speed = request.form.get("speed", default=2000, type=int)
    return render_template("page.html", graph=graph(speed), speed=speed)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.environ.get("PORT", 8000), debug=True)
