import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: Feel free to try out your own images here by changing img_path
# to a file path to another image on your computer!
img_path = "./udacity_sdc.png"

# load color image
bgr_img = cv2.imread(img_path)
# convert to grayscale
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

# normalize, rescale entries to lie in [0,1]
gray_img = gray_img.astype("float32") / 255

# plot image
# plt.imshow(gray_img, cmap='gray')
# plt.show()

# ███████ ██ ██      ████████ ███████ ██████  ███████ 
# ██      ██ ██         ██    ██      ██   ██ ██      
# █████   ██ ██         ██    █████   ██████  ███████ 
# ██      ██ ██         ██    ██      ██   ██      ██ 
# ██      ██ ███████    ██    ███████ ██   ██ ███████ 

filter_vals = np.array([
    [-1, -1, 1, 1], 
    [-1, -1, 1, 1], 
    [-1, -1, 1, 1], 
    [-1, -1, 1, 1]])

print("Filter shape: ", filter_vals.shape)

# define four filters
filter_1 = filter_vals
filter_2 = -filter_1
filter_3 = filter_1.T # .T=transpose
filter_4 = -filter_3 
filters = np.array([filter_1, filter_2, filter_3, filter_4])

# print('Filter 1: \n', filter_1) # Left line Negatives
# print('Filter 2: \n', filter_2) # Right line negatives
# print('Filter 3: \n', filter_3) # Top line Negatives
# print('Filter 4: \n', filter_4) # Bot line Negatives

# visualize all four filters
show_filters = False 
if show_filters:
    fig = plt.figure(figsize=(10, 5))
    for i in range(4):
        ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i], cmap='gray')
        ax.set_title('Filter %s' % str(i+1))
        width, height = filters[i].shape
        for x in range(width):
            for y in range(height):
                ax.annotate(str(filters[i][x][y]), xy=(y,x),
                            horizontalalignment='center',
                            verticalalignment='center',
                            color='white' if filters[i][x][y]<0 else 'black')
# plt.imshow(gray_img, cmap='gray')
# plt.show()
# print(x)

# ███    ██ ███████ ████████ 
# ████   ██ ██         ██    
# ██ ██  ██ █████      ██    
# ██  ██ ██ ██         ██    
# ██   ████ ███████    ██    

# define a neural network with a single convolutional layer with four filters
class Net(nn.Module):
    
    def __init__(self, weight):
        super(Net, self).__init__()
        # initializes the weights of the convolutional layer to be the weights of the 4 defined filters
        k_height, k_width = weight.shape[2:]
        # assumes there are 4 grayscale filters
        self.conv = nn.Conv2d(1, 4, kernel_size=(k_height, k_width), bias=False)
        self.conv.weight = torch.nn.Parameter(weight)

    def forward(self, x):
        # calculates the output of a convolutional layer
        # pre- and post-activation
        conv_x = self.conv(x)
        activated_x = F.relu(conv_x)
        
        # returns both layers
        return conv_x, activated_x
    
# instantiate the model and set the weights
print("Filter Shape", filters.size, type(filters))
a = torch.from_numpy(filters)
print("Torch ", a.size(), type(a))
b = a.unsqueeze(1)
print("Torch.unsqueeze ",b.size(), type(b))
c = b.type(torch.FloatTensor)
print("Torch.FloatTensor", c.size(), type(c))
weight = c
# weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
model = Net(weight)
print("=====")
# print out the layer in the network
print(model)

def viz_layer(layer, n_filters= 4):
    fig = plt.figure(figsize=(20, 20))
    plt.autoscale(tight=True)
    
    for i in range(n_filters):
        ax = fig.add_subplot(int(n_filters/2), int(n_filters/2), i+1, xticks=[], yticks=[])
        # ax = fig.add_subplot(1, n_filters, i+1, xticks=[], yticks=[])
        # grab layer outputs
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        ax.imshow(np.squeeze(layer[0,i].data.numpy()), cmap='gray')
        ax.set_title('Output %s' % str(i+1))

    # plt.rcParams['savefig.pad_inches'] = 0
    plt.show()


# plot original image
# plt.imshow(gray_img, cmap='gray')
# plt.show()

# visualize all filters
# fig = plt.figure(figsize=(12, 6))
# fig.subplots_adjust(left=0, right=1.5, bottom=0.8, top=1, hspace=0.05, wspace=0.05)
# for i in range(4):
#     ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
#     ax.imshow(filters[i], cmap='gray')
#     ax.set_title('Filter %s' % str(i+1))
# plt.show()   

# convert the image into an input Tensor
gray_img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(1)

# get the convolutional layer (pre and post activation)
conv_layer, activated_layer = model(gray_img_tensor)

# visualize the output of a conv layer
viz_layer(conv_layer)


viz_layer(activated_layer)