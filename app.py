from flask import Flask, render_template, request
from PIL import Image
from io import BytesIO
import base64
import numpy as np

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
    image = image.convert('L') # ensure removing any unwanted channels from the browser which can be in RGB format.
    image.save("drawing.png")
    # print(image.size)
    # print(type(image))
    image_array = np.array(image.resize((28, 28)))/255
    image_array = image_array.reshape(1, 1, 28, 28)
    # print(image_array.shape)
    # print(image_array)
    return "Test"


if __name__ == "__main__":
    app.run(debug=True)

