from flask import Flask, render_template, request, jsonify
from PIL import Image
from io import BytesIO
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from google.cloud import storage
import os

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
    # image.save("drawing.png")
    # print(image.size)
    # print(type(image))
    image_array = np.array(image.resize((28, 28)))/255
    image_array = image_array.reshape(1, 1, 28, 28).astype(np.float32)
    image_array = torch.from_numpy(image_array)
    # print(image_array.shape)
    # print(image_array)
    # print(image_array.dtype)


    ######################### Model Definition:
    class MNIST_CNN(nn.Module):
        def __init__(self):
            super(MNIST_CNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size = 5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size = 5)
            self.map = nn.MaxPool2d(2)
            
            self.fc = nn.Linear(320,10)
        
        def forward(self,x):
            in_size = x.size(0)
            x = F.relu(self.map(self.conv1(x)))
            x = F.relu(self.map(self.conv2(x)))
            x = x.view(in_size,-1) #flatten the tensor
            
            x = self.fc(x)
            
            return F.log_softmax(x)


    #### Loading state_dict from Google Cloud
    model_state_output_path = "./model_state_dict.pt"

    # if statement checks if the model_state file is already downloaded.
    if not os.path.exists(model_state_output_path):
        BUCKET_NAME = "mnist_classification_bucket"
        # Creating Storage client
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)

        # Downloading Model State Dict from GC Bucket to Container Local Environment
        gc_model_state_output_path = "model/model_state_dict.pt"
        bucket.blob(gc_model_state_output_path).download_to_filename(model_state_output_path)
    gc_model_state = torch.load(model_state_output_path)

    model = MNIST_CNN()
    model.load_state_dict(gc_model_state)
    model.eval()
    

    ##### Load state_dict from local folder
    # model = MNIST_CNN()
    # model.load_state_dict(torch.load("./Notebooks/CNN_Model_state_dict.pt"))
    # model.eval()

    label_pred = model(image_array)
    _, class_pred = torch.max(label_pred,1)
    # print(class_pred.item())
    return jsonify({'digit': int(class_pred.item())})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

