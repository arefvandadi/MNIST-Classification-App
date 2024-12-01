from flask import Flask, render_template, request
from PIL import Image
from io import BytesIO
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    class NeuralNet(nn.Module):
        def __init__(self):
            super(NeuralNet, self).__init__()
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

    model = NeuralNet()
    model.load_state_dict(torch.load("./Notebooks/CNN_Model_state_dict.pt"))
    model.eval()

    label_pred = model(image_array)
    _, class_pred = torch.max(label_pred,1)
    print(class_pred.item())
    return "Test"


if __name__ == "__main__":
    app.run(debug=True)

