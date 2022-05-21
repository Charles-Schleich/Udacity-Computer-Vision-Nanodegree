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

# Abstracted everything out to fc_model to make the code for cleanliness
import fc_model

print("Pytorch GPU info")
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))


# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                              transforms.Normalize((0.5,),(0.5,))])

# Download and load the training data
trainset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

image, label = next(iter(trainloader))
helper.imshow(image[0,:]);
# plt.show()

# Create the network, define the criterion and optimizer
model = fc_model.Network(784, 10, [512, 256, 128])
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

fc_model.train(model, trainloader, testloader, criterion, optimizer, epochs=1)


# Saving the model from here 
# Saving the model from here 
# Saving the model from here 
print("Model: ", model)
# These are the weights we care about 
print("Model State Keys:" , model.state_dict().keys())

# Saving model to checkpoint.pth
torch.save(model.state_dict(), 'checkpoint.pth')

# loading state dict, but its not connected to a model yet 
state_dict = torch.load('checkpoint.pth')
# Attach it to a model with same shape
model.load_state_dict(state_dict)
# When you load in a checkpoint you must make sure the model has the same architecture when you 
# load it back in

# Model with a wrong_architecture
# model_bad = fc_model.Network(784, 10, [400, 200, 100])
# This will throw an error because the tensor sizes are wrong!
# model_bad.load_state_dict(state_dict)

# So we save the architecture of the model as well !
checkpoint = {'input_size': 784,
              'output_size': 10,
              'hidden_layers': [each.out_features for each in model.hidden_layers],
              'state_dict': model.state_dict()}

torch.save(checkpoint, 'checkpoint_with_struct.pth')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = fc_model.Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    return model

model_with_structure = load_checkpoint('checkpoint_with_struct.pth')
print(model_with_structure)
