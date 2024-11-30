from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    image_data = request.json["image"]
    print(image_data)
    return "Test"


if __name__ == "__main__":
    app.run(debug=True)

