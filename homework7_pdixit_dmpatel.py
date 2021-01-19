#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.nn.functional as F           # layers, activations and more
import torch.optim as optim               # optimizers e.g. gradient descent, 
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# In[2]:


X_train_MNIST = np.load("mnist_train_images.npy")
small = X_train_MNIST[:500, :]


# In[3]:


import torchvision.datasets as datasets
from torchvision import transforms


# In[4]:


transform = transforms.Compose([transforms.ToTensor()                              ])
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=64, shuffle=True)


# In[8]:


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 250)
        self.fc1a = nn.Linear(250, 200)
        self.fc2_mean = nn.Linear(200, 50)
        self.fc2_std = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, 250)
        self.fc3a = nn.Linear(250, 200)
        self.fc4 = nn.Linear(200, 784)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc1a(x))
        
        x_mean = torch.tanh(self.fc2_mean(x))
        x_std = torch.tanh(self.fc2_std(x))
        std = x_std.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        z = eps.mul(std).add_(x_mean)
        z = torch.tanh(self.fc3(z))
        z = torch.tanh(self.fc3a(z))
        
        x = torch.sigmoid(self.fc4(z))
        return x, x_mean, x_std
    
    def generate(self, z):
        z = torch.tanh(self.fc3(z))
        z = torch.tanh(self.fc3a(z))
        x = torch.sigmoid(self.fc4(z))
        return x

net = Net()
print(net)


# In[26]:


optimizer = optim.Adam(net.parameters(), lr=.001)
for i in tqdm(range(150), desc = "epochs"):
    for data in trainloader:
        images, labels = data
        small_tensor = images.reshape((-1,784))
    #     small_tensor = Variable(torch.from_numpy(images))
        prediction_tensor, x_mean, x_std = net.forward(small_tensor)
        prediction = prediction_tensor.data.numpy()

        # loss function
        #criterion = nn.L1Loss()
        #loss1 = criterion(prediction_tensor, small_tensor)
#         loss1 = F.binary_cross_entropy(prediction_tensor, small_tensor)
#         KLD = -0.5 * torch.sum(1 + x_std - x_mean.pow(2) - x_std.exp())
#     #     KLD = KLD/784
#         loss = loss1+(1*KLD)
        reconstruction_loss = F.binary_cross_entropy(prediction_tensor, small_tensor, reduction='sum')
        KL_divergence_loss = -0.5*torch.sum(1+x_std-x_mean*x_mean-x_std.exp())
        loss = reconstruction_loss + KL_divergence_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
#     print(loss)


# In[27]:


new = Variable(torch.randn(1, 50))
img = net.generate(new).data.numpy()
fig, ax = plt.subplots()
imgplot = ax.imshow(img.reshape(28, -1), cmap='Greys')
fig.colorbar(imgplot)
# plt.savefig("generated.png")

# torch.save(net, "model")
# torch.save(net, f"model")


# In[28]:


# new = Variable(torch.randn(1, 50))
images, labels = next(iter(trainloader))
small_tensor = images.reshape((-1,784))
imageNo = 2
img,a,b = net.forward(small_tensor[imageNo])
img = img.data.numpy()

f, axarr = plt.subplots(1,2)
axarr[0].imshow(img.reshape(28, -1), cmap='Greys')
axarr[1].imshow(small_tensor[imageNo].reshape(28, -1), cmap='Greys')


# fig, ax = plt.subplots()
# imgplot = ax.imshow(img.reshape(28, -1), cmap='Greys')
# fig.colorbar(imgplot)


# In[29]:


images, labels = next(iter(trainloader))
small_tensor = images.reshape((-1,784))
imageNo = 2

img,a,b = net.forward(small_tensor)
img = img.data.numpy()


# In[30]:


img.shape



# In[31]:


def show_images(images: list) -> None:
    n: int = len(images)
    f = plt.figure()
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(n, 1, i + 1)
        plt.imshow(images[i])
        
    plt.show(block=True)


# In[35]:


images, labels = next(iter(trainloader))
small_tensor = images.reshape((-1,784))
img,a,b = net.forward(small_tensor)
img = img
# show_images(img[:3, :].reshape(-1,28,28))
save_image(images.view(64, 1, 28, 28), 'VAE-before-PASS.png')
save_image(img.view(64, 1, 28, 28), 'VAE-after-PASS.png')


# In[34]:


new = Variable(torch.randn(64, 50))
img = net.generate(new).data.numpy()
show_images(img[:3, :].reshape(-1,28,28))


# In[20]:


from torchvision.utils import save_image

with torch.no_grad():
    z = Variable(torch.randn(64, 50))
    sample = net.generate(new)
    save_image(sample.reshape(64, 1, 28, 28), 'VAE-AUTO-GENERATED.png')


# In[36]:


mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=64, shuffle=True)


# In[37]:


images, labels = next(iter(testloader))
small_tensor = images.reshape((-1,784))
img,a,b = net.forward(small_tensor)
img = img
# show_images(img[:3, :].reshape(-1,28,28))
save_image(images.view(64, 1, 28, 28), 'VAE-before-PASS.png')
save_image(img.view(64, 1, 28, 28), 'VAE-after-PASS.png')


# In[ ]:




