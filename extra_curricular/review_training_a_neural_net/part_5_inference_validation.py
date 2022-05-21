# Better Generalizing Neural Networks
#   Test for overfitting while training 
# Train + Test Sets help generalize

import numpy as np
import torch 
import matplotlib.pyplot as plt
# torchvision allows us to download and use existing datasets
from torchvision import datasets, transforms
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
import helper 

print("Pytorch GPU info")
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))

# Define Transform to nrmalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
                                transforms.Normalize((0.5), (0.5))])


# Download and load the training data
trainset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


# Arbitrary numbers of hidden layers
hidden_layers = [512,256,128,64]
layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
for each in layer_sizes:
    print(each)

# hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
# Add hidden layers to the ModuleList
# hidden_layers.extend([nn.Linear(h1,h2) for h1, h2 in layer_sizes])

class Network(nn.Module):
    def __init__(self, input_size,output_size,hidden_layers,drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Args
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''

        super().__init__()
        # Add the first layer, input to a hidden layer 
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1,h2) for h1,h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_p)
    def forward(self,x):
        '''Forward pass through the network, returns the output logits '''
        # forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        x=self.output(x)

        return F.log_softmax(x, dim=1)


# if using log-softmax Then use negative log loss as criteron 
# nn.NLLLoss() https://pytorch.org/docs/master/nn.html#nllloss
# use Adam optimizer, includes momentum and trains fasters than SGD

# I can also Can turn off dropout during evaluation 
# model.train() = dropout is on 
# model.eval() = dropout is off
# dropout turns off connections to make sure all the network weights are trained, 
# and that certain connections arent biased

model = Network(784, 10, [516, 256], drop_p=0.5)
criterion = nn.NLLLoss() # negative log loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Implement a function for the validation pass
def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:

        images.resize_(images.shape[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        # returns new tensor with exponential of the elements of the input tensor
        # yi = e^xi 
        ps = torch.exp(output)

        equality = (labels.data == ps.max(dim=1)[1])
        # converts bytes tensor to float tensor 
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy



epochs = 2
steps = 0
running_loss = 0
print_every = 40
for e in range(epochs):
    model.train()
    for images, labels in trainloader:
        steps += 1
        
        # Flatten images into a 784 long vector
        images.resize_(images.size()[0], 784)
        
        optimizer.zero_grad()
        
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            # Make sure network is in eval mode for inference
            model.eval()
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                test_loss, accuracy = validation(model, testloader, criterion)
                
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
            
            running_loss = 0
            
            # Make sure training is back on
            model.train()

#  Testing my Network
model.eval()

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
# Convert 2D image to 1D vector
img = img.view(1, 784)

# Calculate the class probabilities (softmax) for img
with torch.no_grad():
    output = model.forward(img)

ps = torch.exp(output)

# Plot the image and probabilities
helper.view_classify(img.view(1, 28, 28), ps, version='Fashion')
plt.show()
