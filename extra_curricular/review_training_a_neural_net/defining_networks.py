import numpy as np
import torch 

import helper 
import matplotlib.pyplot as plt

# torchvision allows us to download and use existing datasets
from torchvision import datasets, transforms
from torch import nn
from torch import optim
import torch.nn.functional as F


# Define transformer to normalize data 
transform = transforms.Compose(
    [
    # converts to tensor
    transforms.ToTensor() 
    # normalize grayscale images from f0.0 -> f1.0 to f-1.0 -> f1.0
    # subtracts 0.5 from each subpixel value, then divide by 0.5 (multiply by 2)
    ,transforms.Normalize((0.5,),(0.5,)) 
    # ,transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) 
    ])

# Download and load the training data
trainset = datasets.MNIST('MNIST_data/',download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=64, shuffle=True)

# Download and load the training data
testset = datasets.MNIST('MNIST_data/',download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images[1].size())
# plt.imshow(images[1].numpy().squeeze(), cmap="Greys_r")
# plt.show()

# class named network, subclass of nn.Module
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        #Define the layers, 128,64,10 units each
        self.fc1 = nn.Linear(784,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)

    # All clases that inherit from nn.Module Need to have a forward function
    # Expecting that x is a tensor
    def forward(self, x):
        # Passing tensor x through first layer, linear operation 
        # Outputs another tensor
        x = self.fc1(x)
        x = F.relu(x) #apply relu activation function to the above layer
        # Second layer
        x = self.fc2(x)
        x = F.relu(x) # Second relu
        # Third layer
        x = self.fc3(x) # x.dim() = (64,10)
        x = F.softmax(x, dim=1) # Then finally softmax function
        return x 

model = Network()
print(model)

# Initializing weights and biases
print(model.fc1.weight)
print(model.fc1.bias)

# If we want to reinitialize bias and weights
model.fc1.bias.data.fill_(0)
print(model.fc1.bias)

# Initalize the weights with a normal distribution stdev 0.01
model.fc1.weight.data.normal_(std=0.01)  
print(model.fc1.weight)

# Forward pass
# Now that we have a network, let's see what happens when we pass in an image. 
# This is called the forward pass. We're going to convert the image data into a tensor, 
# then pass it through the operations defined by the network architecture.

# trainloader returns a generator, to turn into an iterator call iter(trainloader)
# to get next elem next(iter)
# images.resize_(64,1,784)
images, labels = next(iter(trainloader))
images.resize_(images.shape[0],1,784)
ps = model.forward(images[0]) # ps = probabilites
# helper.view_classify(images[0].view(1,28,28),ps)
# plt.show()

####################################################################################################
# More convenient way to building a model 
# hyper parameters are parameters that define the architecture of the network
input_size = 784
hidden_size = [128,64]
output_size = 10
model = nn.Sequential(nn.Linear(input_size, hidden_size[0])
                     ,nn.ReLU()
                     ,nn.Linear(hidden_size[0], hidden_size[1])
                     ,nn.ReLU()
                     ,nn.Linear(hidden_size[1], output_size)
                     ,nn.Softmax(dim=1)
)
print(model)
images2, labels = next(iter(trainloader))
images2.resize_(images2.shape[0],1,784)
ps = model.forward(images2[0,:]) # ps = probabilites
# helper.view_classify(images2[0].view(1,28,28),ps)
# plt.show()

####################################################################################################
# Another way to create a network is to create an ordered dictionary 
from collections import OrderedDict
# Keys have to be unique ! 
# Keys = names of the layers 
# Values = operations of layers
model = nn.Sequential(OrderedDict([
    ("fc1", nn.Linear(input_size, hidden_size[0])),
    ("relu1", nn.ReLU()),
    ("fc2", nn.Linear(hidden_size[0], hidden_size[1])),
    ("relu2", nn.ReLU()),
    ("output", nn.Linear(hidden_size[1], output_size)),
    ("softmax", nn.Softmax(dim=1)),
]))

print("OrderedDict Version : ",model)



####################################################################################################
###CHARLIE NET 
####################################################################################################
####################################################################################################
# Build a network to classify the MNIST images with three hidden layers. 
# Use 400 units in the first hidden layer, 
# 200 units in the second layer, and 
# 100 units in the third layer. 
# Each hidden layer should have a ReLU activation function, 
# and use softmax on the output layer.

input_size = 784
hidden_size = [400,200,100]
output_size = 10
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
    ("fc1", nn.Linear(input_size, hidden_size[0])),
    ("relu1", nn.ReLU()),
    ("fc2", nn.Linear(hidden_size[0], hidden_size[1])),
    ("relu2", nn.ReLU()),
    ("fc3", nn.Linear(hidden_size[1], hidden_size[2])),
    ("relu3", nn.ReLU()),
    ("output", nn.Linear(hidden_size[2], output_size)),
    ("softmax", nn.Softmax(dim=1)),
]))

print("Charlie net Version : ",model)


images, labels = next(iter(trainloader))
images.resize_(images.shape[0], 1, 784)
ps = model.forward(images[0,:])
helper.view_classify(images[0].view(1, 28, 28), ps)
plt.show()