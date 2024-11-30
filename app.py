from flask import Flask, render_template, request
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    image_data = request.json["image"]
    image_dataUrl = image_data.split(",")[1]
    # print(image_dataUrl)
    image_decoded = base64.b64decode(image_dataUrl)
    image = Image.open(BytesIO(image_decoded))
    image.save("drawing.png")
    return "Test"


if __name__ == "__main__":
    app.run(debug=True)

