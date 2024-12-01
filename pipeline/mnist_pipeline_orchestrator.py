# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
# from kfp.dsl import component, pipeline, Dataset, Model
from kfp import dsl, compiler
from kfp.dsl import OutputPath, InputPath
# from google.cloud import storage

BUCKET_NAME = "mnist_classification_bucket"


#hyper parameters
# input_size = 28*28  #because you will see down the line that our images have the size 28X28
# hidden_size = 100
# num_classes = 10 #becasue our model need to recognize digits from 0 to 9 adding up to 10 classes
# num_epochs = 2
# batch_size = 100
# learning_rate = 0.001


############## Loading Data Component ###############
@dsl.component(
        packages_to_install=["torch", "torchvision", "google-cloud-storage", "kfp"]
)
def load_data(gc_train_dataset_path: OutputPath("Dataset"), gc_test_dataset_path: OutputPath("Dataset")): # type: ignore

    import torch
    import torchvision
    import torchvision.transforms as transforms
    from google.cloud import storage

    BUCKET_NAME = "mnist_classification_bucket"

    #Let's import the famous MNIST data
    #torchvision.transforms.ToTensor --> Convert a PIL Image or numpy.ndarray to tensor
    train_dataset = torchvision.datasets.MNIST(root = './data',train = True, transform = transforms.ToTensor(), download = True)
    test_dataset = torchvision.datasets.MNIST(root = './data',train = False, transform = transforms.ToTensor())

    train_data_path = "./train_data.pt"
    test_data_path = "./test_data.pt"

    torch.save(train_dataset, train_data_path)
    torch.save(test_dataset, test_data_path)
    # torch.save({"data": train_dataset.data, "targets": train_dataset.targets}, train_data_path)
    # torch.save({"data": test_dataset.data, "targets": test_dataset.targets}, test_data_path)


    # Creating Storage client and uploading datasets to GC
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    gc_train_dataset_destination_blob_path = "datasets/train_data.pt"
    # train_blob = bucket.blob(gc_train_dataset_destination_blob_path)
    # train_blob.upload_from_file(train_data_path)
    with open(train_data_path, "rb") as train_file:
        bucket.blob(gc_train_dataset_destination_blob_path).upload_from_file(train_file)

    gc_test_dataset_destination_blob_path = "datasets/test_data.pt"
    # test_blob = bucket.blob(gc_test_dataset_destination_blob_path)
    # test_blob.upload_from_file(test_data_path)
    with open(test_data_path, "rb") as test_file:
        bucket.blob(gc_test_dataset_destination_blob_path).upload_from_file(test_file)

    # gc_train_dataset_path = f"gs://{BUCKET_NAME}/{gc_train_dataset_destination_blob_path}"
    # gc_test_dataset_path = f"gs://{BUCKET_NAME}/{gc_test_dataset_destination_blob_path}"

    with open(gc_train_dataset_path, "w") as train_output:
        train_output.write(f"gs://{BUCKET_NAME}/{gc_train_dataset_destination_blob_path}")
    
    with open(gc_test_dataset_path, "w") as test_output:
        test_output.write(f"gs://{BUCKET_NAME}/{gc_test_dataset_destination_blob_path}")
    # return gc_train_dataset_path, gc_test_dataset_path



############## Trainer Component ###############
@dsl.component(
        packages_to_install=["torch", "torchvision", "google-cloud-storage", "numpy"]
)
def trainer(gc_train_dataset_path: InputPath("Dataset"), batch_size: int, learning_rate: float, num_epochs: int):# type: ignore
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from google.cloud import storage

    ##############   CNN Model Definition   ###############
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


    model = MNIST_CNN()
    model.train()


    BUCKET_NAME = "mnist_classification_bucket"
    # Creating Storage client and uploading datasets to GC
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    # train_dataset = torch.load(gc_train_dataset_path, weights_only=True)
    bucket.blob("datasets/train_data.pt").download_to_filename("./train_data.pt")
    train_data = torch.load("./train_data.pt")

    train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
    # train_dataset = torch.utils.data.TensorDataset(train_data["data"], train_data["targets"])
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)


    #Loss Definition. We are going to use Cross Entropy Loss which will include Softmax activation function to output possibilities
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

    # batch_num = len(train_loader)

    for epoch in range(num_epochs):
        for i, (samples, labels) in enumerate(train_loader):
            
            #Let's reshape samples from 4 dimension to 2 dim, each row having 28*28 elements for each image
            #samples = samples.reshape(-1,28*28).to(device)
            
            #Forward pass
            label_pred = model(samples)
            
            #Loss
            Loss = criterion(label_pred, labels)
            
            #Zero grads
            optimizer.zero_grad()
            
            #backward
            Loss.backward()
            
            #updating parameters
            optimizer.step()
            
        print("epoch = {} / {}  :  Loss = {Lossvalue:.4f}".format(epoch+1, num_epochs, Lossvalue = Loss.item()))

    
    ### Save it first in the containerized environment, and then upload it to GC

    model_state_output_path = "./model_state_dict.pt"
    torch.save(model.state_dict(), model_state_output_path)

    gc_model_state_output_path = "model/model_state_dict.pt"
    with open(model_state_output_path, "rb") as model_state:
        bucket.blob(gc_model_state_output_path).upload_from_file(model_state)


############## Evaluation Component ###############
@dsl.component(
        packages_to_install=["torch", "torchvision", "google-cloud-storage", "numpy"]
)
def evaluator(gc_test_dataset_path: InputPath("Dataset"), accuracy_output: OutputPath("Metrics"), batch_size: int):# type: ignore
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from google.cloud import storage


    ##############   CNN Model Definition   ###############
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
    

    BUCKET_NAME = "mnist_classification_bucket"
    # Creating Storage client
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    # Downloading Model State Dict from GC Bucket to Container Local Environment
    gc_model_state_output_path = "model/model_state_dict.pt"
    model_state_output_path = "./model_state_dict.pt"
    bucket.blob(gc_model_state_output_path).download_to_filename(model_state_output_path)
    model_state = torch.load(model_state_output_path)
    
    # Load the downloaded model state into the model
    model = MNIST_CNN()
    model.load_state_dict(model_state)
    model.eval()

    # Downloading Test Dataset from GC Bucket to Container Local Environment
    bucket.blob("datasets/test_data.pt").download_to_filename("./test_data.pt")
    test_data = torch.load("./test_data.pt")
    test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False)


    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for i, (samples, labels) in enumerate(test_loader):
            
            #Let's reshape test samples from 4 dimension to 2 dim, each row having 28*28 elements for each image
            #samples = samples.reshape(-1,28*28).to(device)
            # labels = labels.to(device)
            
            #test samples predicted results using the train Model
            test_label_pred = model(samples)
            
            #Notice that we are not using softmax here to get the probabilities. The biggest value has the more probability... 
            #...and is the predicted class
            #Also torch.max() returns the max value and the index for max value. We only need the index whih represents the class
            _, class_pred = torch.max(test_label_pred,1)  # number 1 in torch.max method means max along the rows. 0 means columns
            n_samples += labels.shape[0]
            n_correct += (class_pred == labels).sum().item()
        
        print('\n')
        print('Number of Tested Samples      = ',n_samples)
        print('Number of Correct Predictions = ',n_correct)
        acc = 100 * n_correct/n_samples
        print("Accuracy = {Accuracy:.2f}%".format(Accuracy = acc))

    # Save accuracy to output
    local_accuracy_file_path = "./accuracy.txt"
    with open(local_accuracy_file_path, "w") as f:
        f.write(f"accuracy: {acc:.2f}")

    # Upload accuracy file to the bucket
    accuracy_blob_path = "model/accuracy.txt"  # Define where to save in GCS
    bucket.blob(accuracy_blob_path).upload_from_filename(local_accuracy_file_path)

    # Save the path to the accuracy file in the output
    with open(accuracy_output, "w") as f:
        f.write(f"gs://{BUCKET_NAME}/{accuracy_blob_path}")



@dsl.pipeline
def mnist_pipeline():
    load_data_op = load_data()

    trainer(gc_train_dataset_path=load_data_op.outputs["gc_train_dataset_path"] , batch_size=100, learning_rate=0.001, num_epochs=2)

    evaluator(gc_test_dataset_path=load_data_op.outputs["gc_test_dataset_path"], batch_size=100)


compiler.Compiler().compile(pipeline_func=mnist_pipeline, package_path="mnist_pipeline.yaml")

# Execution on Google Cloud
from google.cloud import aiplatform
from google.cloud.aiplatform import PipelineJob

# Set your Google Cloud Project ID and region
PROJECT_ID = "mnist-classification-app"
REGION = "us-east5"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline-root/"  # Replace with your GCS bucket

aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=PIPELINE_ROOT,
    service_account="mnist-classification-app@mnist-classification-app.iam.gserviceaccount.com"
)

job = PipelineJob(
        ### path of the yaml file to execute
        template_path="./mnist_pipeline.yaml",
        ### name of the pipeline
        display_name=f"mnist_classification_pipeline",
        ### pipeline arguments (inputs)
        ### {"recipient": "World!"} for this example
        # parameter_values=pipeline_arguments,
        ### region of execution
        # location="us-east5",
        ### root is where temporary files are being 
        ### stored by the execution engine
        pipeline_root=PIPELINE_ROOT,
)

### submit for execution
job.submit()

### check to see the status of the job
job.state

