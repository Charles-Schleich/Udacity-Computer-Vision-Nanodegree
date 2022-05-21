# Low key worried bots are gonna flag this
# https://www.  _yt_  .com/watch?v=u8hDj5aJK6I

import numpy as np
import torch 

import helper 
import matplotlib.pyplot as plt

# torchvision allows us to download and use existing datasets
from torchvision import datasets, transforms
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict

print("Start")
# requires_grad tells pytorch to track all operations on this tensor using autograd
x = torch.randn(2,2, requires_grad=True)
print(x)

# 
y = x**2 
print("y",y)
print("y",y.grad_fn) # gives operation used to create this tensor 

z = y.mean()
print("z",z)
print("z",z.grad_fn)

# nothing here as have not done a backwards pass 
print("x",x.grad)
z.backward()
print("x",x.grad)
print("x",x/2)

# We also need to define the optimizer we're using, SGD or Adam, or something
# along those lines. Here I'll just use SGD with `torch.optim.SGD`, 
# passing in the network parameters and the learning rate.

from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                            #   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                              transforms.Normalize((0.5,),(0.5,)),
                             ])
# Download and load the training data
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Hyperparameters for our network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10


# I'll build a network with nn.Sequential here. Only difference from the last part 
# is I'm not actually using softmax on the output, but instead just using the raw 
# output from the last layer. This is because the output from softmax is a 
# probability distribution. Often, the output will have values really close to 
# zero or really close to one. Due to inaccuracies with representing numbers as 
# floating points, computations with a softmax output can lose accuracy and become 
# unstable. To get around this, we'll use the raw output, called the logits, to 
# calculate the loss.

# Build a feed-forward network
model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                      ('relu2', nn.ReLU()),
                      ('logits', nn.Linear(hidden_sizes[1], output_size))]))


########################################################################################################################
##### Example of single run through the network
########################################################################################################################
criterion = nn.CrossEntropyLoss();
# Optimizer with Learning rate of 0.01 
optimizer = optim.SGD(model.parameters(), lr=0.01)
# First, let's consider just one learning step before looping through all the data. The general process with PyTorch:
#   - Make a forward pass through the network to get the logits
#   - Use the logits to calculate the loss
#   - Perform a backward pass through the network with loss.backward() to calculate the gradients
#   - Take a step with the optimizer to update the weights

print("Before ", model.fc1.weight)
images, labels = next(iter(trainloader))
images.resize_(64, 784)
# optimizer.zero_grad : Zeros out all gradients that are on tensors that are being trained
# reason every time you do loss.backward() it accumulates the gradients 
# So each time you do it will accumulate the gradients, we DONT want this. 
# we want to calculate new gradients each time do a backwards_prop
# So on each iteration we will clear grads 
optimizer.zero_grad()
# Forward pass
output = model.forward(images)
loss = criterion(output, labels)
# backwards pass
loss.backward()
print("Gradient - ",  model.fc1.weight.grad )
# do optimizer step
optimizer.step()
print("updated weights - ",  model.fc1.weight )
########################################################################################################################
##### Multiple loops of training! 
########################################################################################################################

epochs = 5
print_every = 40
steps =0 
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
# images, labels = next(iter(trainloader))


img = images[0].view(1,784)
with torch.no_grad():
    logits = model.forward(img)

ps = F.softmax(logits, dim=1)
helper.view_classify(img.view(1, 28, 28), ps)
plt.show()
