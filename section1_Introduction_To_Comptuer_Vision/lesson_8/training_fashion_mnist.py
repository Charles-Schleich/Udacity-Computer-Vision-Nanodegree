import torch
import torchvision

from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from torchvision.datasets import FashionMNIST
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt


data_transform = transforms.ToTensor()

# choose the training and test datasets
train_data = FashionMNIST(
    root="../../extra_curricular/review_training_a_neural_net/F_MNIST_data/",
    train=True,
    download=True,
    transform=data_transform,
)

test_data = FashionMNIST(
    root="../../extra_curricular/review_training_a_neural_net/F_MNIST_data/",
    train=True,
    download=True,
    transform=data_transform,
)


# Print out some stats about the training and test data
print("Train data, number of images: ", len(train_data))
print("Test data, number of images: ", len(test_data))


batch_size = 20

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# specify the image classes
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Visualise some training data
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

# Numpy numpy.ndarray size / shape
print("Image Resolution:", np.squeeze(images[0]).shape)


# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(batch_size):
    # fig.add_subplot(rows, ncols, index)
    ax = fig.add_subplot(
        batch_size // 5, batch_size // 4, idx + 1, xticks=[], yticks=[]
    )
    # ax = fig.add_subplot(2, batch_size//2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap="gray")
    ax.set_title(classes[labels[idx]])
# plt.show()


class Net(nn.Module):
    def __init__(self, drop_p=0.1):
        super(Net, self).__init__()

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        # Convolutional filters weights are randomly generated - no manual generation or invocation

        # 1 input image channel (Grayscale image)
        # 10 output channels / feature maps
        # 3x3 conv kernel
        self.conv1 = nn.Conv2d(1, 10, 3)

        # Maxpool, Kernel Size = 2, Stride = 2 
        # Therefore no overlapping
        self.pool = nn.MaxPool2d(2, 2)

        # Second Conv2D layer
        # 1 input image channel, 20 output channels / feature maps, 3x3 conv kernel
        self.conv2 = nn.Conv2d(10, 20, 3)

        # Fully-connected layers, also known as linear layers,
        # torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
        # Conv2 output is an image of 11 x 11 
        # Max pooling with stride = 2 means the input to the linear layer is is 5 x 5 
        # Therefor, 20 * 5 * 5 
        self.linear1 = nn.Linear(20*5*5, 10)

        # DROPOUT LAYERS
        # https://stats.stackexchange.com/questions/240305/where-should-i-place-dropout-layers-in-a-neural-network
        self.dropout = nn.Dropout(p=drop_p)
        # You must place any layers with trainable weights, such as convolutional layers, in the __init__ function and refer to them in the forward functio;
        # any layers or functions that always behave in the same way, such as a pre-defined activation function, may appear in either the __init__ or the forward function.
        # In practice, you'll often see conv/pool layers defined in __init__ and activations defined in forward.

    def forward(self, x):
        # CONV->RELU->DROP->POOL

        # x -> conv1 -> relu -> pool -> y
        # print("x: ", x.size())  # torch.Size([20, 1, 28, 28])
        relu1_out = F.relu(self.conv1(x))
        # drop1_out = self.dropout(relu1_out)
        # print("relu1_out: ", relu1_out.size())  # torch.Size([20, 10, 26, 26])
        pool1_out = self.pool(relu1_out)
        # print("pool1_out: ", pool1_out.size())  # torch.Size([20, 10, 13, 13])


        relu2_out = F.relu(self.conv2(pool1_out))
        # drop2_out = self.dropout(relu2_out)
        pool2_out = self.pool(relu2_out)
        # print("pool2_out: ", pool2_out.size())  # torch.Size([20, 10, 13,13])

        flatten_out = pool2_out.view(x.size(0), -1)
        # print("flatten_out: ", flatten_out.size())  # torch.Size([20, 10, 13,13])
        linear1_out = F.relu(self.linear1(flatten_out))



        softmax_out = F.log_softmax(linear1_out, dim=1)
        return softmax_out


net = Net()
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# ██████  ██████  ███████     ████████ ██████   █████  ██ ███    ██ ███████ ██████      
# ██   ██ ██   ██ ██             ██    ██   ██ ██   ██ ██ ████   ██ ██      ██   ██     
# ██████  ██████  █████          ██    ██████  ███████ ██ ██ ██  ██ █████   ██   ██     
# ██      ██   ██ ██             ██    ██   ██ ██   ██ ██ ██  ██ ██ ██      ██   ██     
# ██      ██   ██ ███████        ██    ██   ██ ██   ██ ██ ██   ████ ███████ ██████      

# PRE Trained Accuracy
# Calculate accuracy before training
correct = 0
total = 0
# Iterate through test dataset
for images, labels in test_loader:

    # warp input images in a Variable wrapper
    images = Variable(images)

    # forward pass to get outputs
    # the outputs are a series of class scores
    outputs = net(images)

    # get the predicted class from the maximum value in the output-list of class scores
    _, predicted = torch.max(outputs.data, 1)

    # count up total number of correct labels
    # for which the predicted and true labels are equal
    total += labels.size(0)
    correct += (predicted == labels).sum()
# calculate the accuracy
# to convert `correct` from a Tensor into a scalar, use .item()
accuracy = 100 * correct.item() / total

# print it out!
print("Accuracy before training: ", accuracy)
