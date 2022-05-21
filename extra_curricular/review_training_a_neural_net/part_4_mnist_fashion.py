import matplotlib.pyplot as plt
import numpy as np
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from collections import OrderedDict

import helper



# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                transforms.Normalize((0.5), (0.5))]
                                
                                )
# Download and load the training data
trainset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

image, label = next(iter(trainloader))
helper.imshow(image[0,:]);

# TODO: Define your network architecture here
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                      ('relu2', nn.ReLU()),
                      ('logits', nn.Linear(hidden_sizes[1], output_size))]))

# TODO: Create the network, define the criterion and optimizer
criterion = nn.CrossEntropyLoss();
# Optimizer with Learning rate of 0.01 
optimizer = optim.SGD(model.parameters(), lr=0.01)

# TODO: Train the network here
# Test out your network!
epochs = 5
print_every = 40
steps = 0 
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        steps+=1 
        images.resize_(images.size()[0], 784)

        optimizer.zero_grad()
        # Forward pass
        output = model.forward(images)
        loss = criterion(output, labels)
        # backwards pass
        loss.backward()
        # weight update step
        optimizer.step()
        # Loss is a scalar tensor, to get the value out of the tensor
        running_loss += loss.item()
        if steps % print_every ==0:
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every)
                )
            running_loss = 0
# Training is finished training
images, labels = next(iter(trainloader))

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
# Convert 2D image to 1D vector
img = img.resize_(1, 784)

# TODO: Calculate the class probabilities (softmax) for img
# ps = 

img = images[0].view(1,784)
with torch.no_grad():
    logits = model.forward(img)

ps = F.softmax(logits, dim=1)
# helper.view_classify(img.view(1, 28, 28), ps)
# Plot the image and probabilities
helper.view_classify(img.resize_(1, 28, 28), ps, version='Fashion')
plt.show()

