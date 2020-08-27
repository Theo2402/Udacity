# Udacity


Face Generation
In this project, you'll define and train a DCGAN on a dataset of faces. Your goal is to get a generator network to generate new images of faces that look as realistic as possible!
The project will be broken down into a series of tasks from loading in data to defining and training adversarial networks. At the end of the notebook, you'll be able to visualize the results of your trained Generator to see how it performs; your generated samples should look like fairly realistic faces with small amounts of noise.
Get the Data
You'll be using the CelebFaces Attributes Dataset (CelebA) to train your adversarial networks.
This dataset is more complex than the number datasets (like MNIST or SVHN) you've been working with, and so, you should prepare to define deeper networks and train them for a longer time to get good results. It is suggested that you utilize a GPU for training.
Pre-processed Data
Since the project's main focus is on building the GANs, we've done some of the pre-processing for you. Each of the CelebA images has been cropped to remove parts of the image that don't include a face, then resized down to 64x64x3 NumPy images. Some sample data is show below.

If you are working locally, you can download this data by clicking here
This is a zip file that you'll need to extract in the home directory of this notebook for further loading and processing. After extracting the data, you should be left with a directory of data processed_celeba_small/
In [1]:
# can comment out after executing
#!unzip processed_celeba_small.zip
In [2]:
data_dir = 'processed_celeba_small/'

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import problem_unittests as tests
#import helper

%matplotlib inline
Visualize the CelebA Data
The CelebA dataset contains over 200,000 celebrity images with annotations. Since you're going to be generating faces, you won't need the annotations, you'll only need the images. Note that these are color images with 3 color channels (RGB)#RGB_Images) each.
Pre-process and Load the Data
Since the project's main focus is on building the GANs, we've done some of the pre-processing for you. Each of the CelebA images has been cropped to remove parts of the image that don't include a face, then resized down to 64x64x3 NumPy images. This pre-processed dataset is a smaller subset of the very large CelebA data.
There are a few other steps that you'll need to transform this data and create a DataLoader.
Exercise: Complete the following get_dataloader function, such that it satisfies these requirements:
Your images should be square, Tensor images of size image_size x image_size in the x and y dimension.
Your function should return a DataLoader that shuffles and batches these Tensor images.
ImageFolder
To create a dataset given a directory of images, it's recommended that you use PyTorch's ImageFolder wrapper, with a root directory processed_celeba_small/ and data transformation passed in.
In [3]:
# necessary imports
import torch
from torchvision import datasets
from torchvision import transforms
In [4]:
def get_dataloader(batch_size, image_size, data_dir='processed_celeba_small/'):
    """
    Batch the neural network data using DataLoader
    :param batch_size: The size of each batch; the number of images in a batch
    :param img_size: The square size of the image data (x, y)
    :param data_dir: Directory where image data is located
    :return: DataLoader with batched data
    """
    
    # TODO: Implement function and return a dataloader
    transform = transforms.Compose([transforms.Resize(image_size),
                                   transforms.ToTensor()]) 
    image = datasets.ImageFolder(data_dir, transform=transform)
    
    DataLoader = torch.utils.data.DataLoader( image , shuffle=True, batch_size=batch_size)
    
    return DataLoader
Create a DataLoader
Exercise: Create a DataLoader celeba_train_loader with appropriate hyperparameters.
Call the above function and create a dataloader to view images.
You can decide on any reasonable batch_size parameter
Your image_size must be 32. Resizing the data to a smaller size will make for faster training, while still creating convincing images of faces!
In [5]:
# Define function hyperparameters
batch_size = 100
img_size = 32

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# Call your function and get a dataloader
celeba_train_loader = get_dataloader(batch_size, img_size)
Next, you can view some images! You should seen square images of somewhat-centered faces.
Note: You'll need to convert the Tensor images into a NumPy type and transpose the dimensions to correctly display an image, suggested imshow code is below, but it may not be perfect.
In [6]:
# helper display function
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# obtain one batch of training images
dataiter = iter(celeba_train_loader)
images, _ = dataiter.next() # _ for no labels

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(20, 4))
plot_size=20
for idx in np.arange(plot_size):
    ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])

Exercise: Pre-process your image data and scale it to a pixel range of -1 to 1
You need to do a bit of pre-processing; you know that the output of a tanh activated generator will contain pixel values in a range from -1 to 1, and so, we need to rescale our training images to a range of -1 to 1. (Right now, they are in a range from 0-1.)
In [7]:
# TODO: Complete the scale function
def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1. 
       This function assumes that the input x is already scaled from 0-1.'''
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    
    min, max = feature_range
    x = x * (max - min) + min
    
    return x
In [8]:
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# check scaled range
# should be close to -1 to 1
img = images[0]
scaled_img = scale(img)

print('Min: ', scaled_img.min())
print('Max: ', scaled_img.max())
Min:  tensor(-1.)
Max:  tensor(0.7098)
Define the Model
A GAN is comprised of two adversarial networks, a discriminator and a generator.
Discriminator
Your first task will be to define the discriminator. This is a convolutional classifier like you've built before, only without any maxpooling layers. To deal with this complex data, it's suggested you use a deep network with normalization. You are also allowed to create any helper functions that may be useful.
Exercise: Complete the Discriminator class
The inputs to the discriminator are 32x32x3 tensor images
The output should be a single value that will indicate whether a given image is real or fake
In [9]:
import torch.nn as nn
import torch.nn.functional as F


# helper
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels,kernel_size, stride, padding, bias=False)
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
     
    return nn.Sequential(*layers)
In [10]:
class Discriminator(nn.Module):

    def __init__(self, conv_dim):
        """
        Initialize the Discriminator Module
        :param conv_dim: The depth of the first convolutional layer
        """
        super(Discriminator, self).__init__()
        
        # complete init function
        self.conv_dim = conv_dim
        
        self.conv1 = conv(3, conv_dim, 4, batch_norm=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4,conv_dim*8,4) 
        
        self.fc = nn.Linear(conv_dim*8*2*2, 1)
        self.Dropout = nn.Dropout(0.2)
        
        

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: Discriminator logits; the output of the neural network
        """
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x),0.2)
        x = self.Dropout(x)
        
        x = x.view(-1, self.conv_dim*8*2*2)
        x = self.fc(x) 
        
        return x


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_discriminator(Discriminator)
Tests Passed
Generator
The generator should upsample an input and generate a new image of the same size as our training data 32x32x3. This should be mostly transpose convolutional layers with normalization applied to the outputs.
Exercise: Complete the Generator class
The inputs to the generator are vectors of some length z_size
The output should be a image of shape 32x32x3
In [11]:
#helper 

def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    layers = []
    
    transpose_conv_layer = nn.ConvTranspose2d(in_channels, out_channels,kernel_size, stride, padding, bias=False)
    layers.append(transpose_conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
        
    return nn.Sequential(*layers)
In [12]:
class Generator(nn.Module):
    
    def __init__(self, z_size, conv_dim):
        """
        Initialize the Generator Module
        :param z_size: The length of the input latent vector, z
        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
        """
        super(Generator, self).__init__()

        # complete init function
        
        self.fc = nn.Linear(z_size, conv_dim*8*2*2)
        self.conv_dim = conv_dim

        self.t_conv1 = deconv(conv_dim*8,conv_dim*4,4)
        self.t_conv2 = deconv(conv_dim*4, conv_dim*2, 4)
        self.t_conv3 = deconv(conv_dim*2, conv_dim, 4)
        self.t_conv4 = deconv(conv_dim, 3, 4, batch_norm=False)
        self.Dropout = nn.Dropout(0.2)


    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: A 32x32x3 Tensor image as output
        """
        # define feedforward behavior
        x = self.fc(x)
        x = x.view(-1, self.conv_dim*8, 2, 2) 
        
        x = F.relu(self.t_conv1(x))
        x = self.Dropout(x)
        x = F.relu(self.t_conv2(x))
        x = self.Dropout(x)
        x = F.relu(self.t_conv3(x))
        x = self.Dropout(x)
        x = self.t_conv4(x)
        
        x = F.tanh(x)
        return x

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_generator(Generator)
Tests Passed
Initialize the weights of your networks
To help your models converge, you should initialize the weights of the convolutional and linear layers in your model. From reading the original DCGAN paper, they say:
All weights were initialized from a zero-centered Normal distribution with standard deviation 0.02.
So, your next task will be to define a weight initialization function that does just this!
You can refer back to the lesson on weight initialization or even consult existing model code, such as that from the networks.py file in CycleGAN Github repository to help you complete this function.
Exercise: Complete the weight initialization function
This should initialize only convolutional and linear layers
Initialize the weights to a normal distribution, centered around 0, with a standard deviation of 0.02.
The bias terms, if they exist, may be left alone or set to 0.
In [13]:
def weights_init_normal(m):
    """
    Applies initial weights to certain layers in a model .
    The weights are taken from a normal distribution 
    with mean = 0, std dev = 0.02.
    :param m: A module or layer in a network    
    """
    # classname will be something like:
    # `Conv`, `BatchNorm2d`, `Linear`, etc.
    classname = m.__class__.__name__
    
    
    # TODO: Apply initial weights to convolutional and linear layers
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    
Build complete network
Define your models' hyperparameters and instantiate the discriminator and generator from the classes defined above. Make sure you've passed in the correct input arguments.
In [14]:
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
def build_network(d_conv_dim, g_conv_dim, z_size):
    # define discriminator and generator
    D = Discriminator(d_conv_dim)
    G = Generator(z_size=z_size, conv_dim=g_conv_dim)

    # initialize model weights
    D.apply(weights_init_normal)
    G.apply(weights_init_normal)

    print(D)
    print()
    print(G)
    
    return D, G
Exercise: Define model hyperparameters
In [15]:
# Define model hyperparams
d_conv_dim = 32
g_conv_dim = 32
z_size = 100

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
D, G = build_network(d_conv_dim, g_conv_dim, z_size)
Discriminator(
  (conv1): Sequential(
    (0): Conv2d(3, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
  )
  (conv2): Sequential(
    (0): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (conv3): Sequential(
    (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (conv4): Sequential(
    (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (fc): Linear(in_features=1024, out_features=1, bias=True)
  (Dropout): Dropout(p=0.2)
)

Generator(
  (fc): Linear(in_features=100, out_features=1024, bias=True)
  (t_conv1): Sequential(
    (0): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (t_conv2): Sequential(
    (0): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (t_conv3): Sequential(
    (0): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (t_conv4): Sequential(
    (0): ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
  )
  (Dropout): Dropout(p=0.2)
)
Training on GPU
Check if you can train on GPU. Here, we'll set this as a boolean variable train_on_gpu. Later, you'll be responsible for making sure that
Models,
Model inputs, and
Loss function arguments
Are moved to GPU, where appropriate.
In [16]:
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import torch

# Check for a GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Training on GPU!')
Training on GPU!
Discriminator and Generator Losses
Now we need to calculate the losses for both types of adversarial networks.
Discriminator Losses
For the discriminator, the total loss is the sum of the losses for real and fake images, d_loss = d_real_loss + d_fake_loss.
Remember that we want the discriminator to output 1 for real images and 0 for fake images, so we need to set up the losses to reflect that.
Generator Loss
The generator loss will look similar only with flipped labels. The generator's goal is to get the discriminator to think its generated images are real.
Exercise: Complete real and fake loss functions
You may choose to use either cross entropy or a least squares error loss to complete the following real_loss and fake_loss functions.
In [17]:
def real_loss(D_out):
    '''Calculates how close discriminator outputs are to being real.
       param, D_out: discriminator logits
       return: real loss'''
    
    batch_size = D_out.size(0)
    labels = torch.ones(batch_size)
    if train_on_gpu:
        labels = labels.cuda()
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss


def fake_loss(D_out):
    '''Calculates how close discriminator outputs are to being fake.
       param, D_out: discriminator logits
       return: fake loss'''
    
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size) 
    if train_on_gpu:
        labels = labels.cuda()
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
   
    return loss
Optimizers
Exercise: Define optimizers for your Discriminator (D) and Generator (G)
Define optimizers for your models with appropriate hyperparameters.
In [18]:
import torch.optim as optim

#hyperparameters
lr = 0.0002
beta1=0.5
beta2=0.999

# Create optimizers for the discriminator D and generator G
d_optimizer = optim.Adam(D.parameters(), lr, [beta1, beta2])
g_optimizer = optim.Adam(G.parameters(), lr, [beta1, beta2])
Training
Training will involve alternating between training the discriminator and the generator. You'll use your functions real_loss and fake_loss to help you calculate the discriminator losses.
You should train the discriminator by alternating on real and fake images
Then the generator, which tries to trick the discriminator and should have an opposing loss function
Saving Samples
You've been given some code to print out some loss statistics and save some generated "fake" samples.
Exercise: Complete the training function
Keep in mind that, if you've moved your models to GPU, you'll also have to move any model inputs to GPU.
In [19]:
def train(D, G, n_epochs, print_every=50):
    '''Trains adversarial networks for some number of epochs
       param, D: the discriminator network
       param, G: the generator network
       param, n_epochs: number of epochs to train for
       param, print_every: when to print and record the models' losses
       return: D and G losses'''
    
    # move models to GPU
    if train_on_gpu:
        D.cuda()
        G.cuda()

    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size=16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()
    # move z to GPU if available
    if train_on_gpu:
        fixed_z = fixed_z.cuda()

    # epoch training loop
    for epoch in range(n_epochs):

        # batch training loop
        for batch_i, (real_images, _) in enumerate(celeba_train_loader):

            batch_size = real_images.size(0)
            real_images = scale(real_images)

            # ===============================================
            #         YOUR CODE HERE: TRAIN THE NETWORKS
            # ===============================================
            
            
            # 1. Train the discriminator on real and fake images    
            
        
            d_optimizer.zero_grad()

            if train_on_gpu:
                 real_images = real_images.cuda()
                
            D_real = D(real_images)
            D_real_loss = real_loss(D_real)
        
            z = np.random.uniform(-1, 1, size = (batch_size, z_size))
            z = torch.from_numpy(z).float()
            if train_on_gpu:
                z = z.cuda()
        
        
        
            fake = G(z)
            D_fake = D(fake)
            D_fake_loss = fake_loss(D_fake)
        

            d_loss = D_real_loss + D_fake_loss
            d_loss.backward()
            d_optimizer.step()

        
        
            # 2. Train the generator with an adversarial loss##   Second: D_Y, real and fake loss components   ##
        
       
            g_optimizer.zero_grad()
        
            z = np.random.uniform(-1, 1, size = (batch_size, z_size))
            z = torch.from_numpy(z).float()
            if train_on_gpu:
                z = z.cuda()
        
            fake = G(z)
            D_fake = D(fake)
            g_loss = real_loss(D_fake)
        
            g_loss.backward()
            g_optimizer.step()
               
            
            # ===============================================
            #              END OF YOUR CODE
            # ===============================================

            # Print some loss stats
            if batch_i % print_every == 0:
                # append discriminator loss and generator loss
                losses.append((d_loss.item(), g_loss.item()))
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch+1, n_epochs, d_loss.item(), g_loss.item()))


        ## AFTER EACH EPOCH##    
        # this code assumes your generator is named G, feel free to change the name
        # generate and save sample, fake images
        G.eval() # for generating samples
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train() # back to training mode

    # Save training generator samples
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)
    
    # finally return losses
    return losses
Set your number of training epochs and train your GAN!
In [20]:
# set number of epochs 
n_epochs = 20


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# call training function
losses = train(D, G, n_epochs=n_epochs)
Epoch [    1/   20] | d_loss: 1.3692 | g_loss: 0.8591
Epoch [    1/   20] | d_loss: 0.0859 | g_loss: 4.1037
Epoch [    1/   20] | d_loss: 0.0404 | g_loss: 4.8349
Epoch [    1/   20] | d_loss: 0.1086 | g_loss: 5.4464
Epoch [    1/   20] | d_loss: 1.5037 | g_loss: 0.6091
Epoch [    1/   20] | d_loss: 2.9289 | g_loss: 4.2639
Epoch [    1/   20] | d_loss: 0.4791 | g_loss: 2.6293
Epoch [    1/   20] | d_loss: 0.6583 | g_loss: 3.1233
Epoch [    1/   20] | d_loss: 0.4047 | g_loss: 2.9672
Epoch [    1/   20] | d_loss: 0.7347 | g_loss: 2.4033
Epoch [    1/   20] | d_loss: 0.7180 | g_loss: 1.9271
Epoch [    1/   20] | d_loss: 0.5478 | g_loss: 2.2539
Epoch [    1/   20] | d_loss: 0.7652 | g_loss: 2.2813
Epoch [    1/   20] | d_loss: 0.6161 | g_loss: 1.9769
Epoch [    1/   20] | d_loss: 0.5670 | g_loss: 3.6510
Epoch [    1/   20] | d_loss: 0.5733 | g_loss: 2.5500
Epoch [    1/   20] | d_loss: 0.8381 | g_loss: 3.6838
Epoch [    1/   20] | d_loss: 0.5198 | g_loss: 1.9887
Epoch [    2/   20] | d_loss: 0.5218 | g_loss: 2.2094
Epoch [    2/   20] | d_loss: 0.6527 | g_loss: 4.6379
Epoch [    2/   20] | d_loss: 0.4898 | g_loss: 3.9323
Epoch [    2/   20] | d_loss: 0.2840 | g_loss: 2.5329
Epoch [    2/   20] | d_loss: 1.5696 | g_loss: 2.6600
Epoch [    2/   20] | d_loss: 0.4994 | g_loss: 3.8181
Epoch [    2/   20] | d_loss: 0.5430 | g_loss: 3.5640
Epoch [    2/   20] | d_loss: 0.4976 | g_loss: 2.2204
Epoch [    2/   20] | d_loss: 0.6858 | g_loss: 4.0512
Epoch [    2/   20] | d_loss: 0.4130 | g_loss: 4.1112
Epoch [    2/   20] | d_loss: 0.3920 | g_loss: 2.8526
Epoch [    2/   20] | d_loss: 0.6090 | g_loss: 1.4879
Epoch [    2/   20] | d_loss: 0.5875 | g_loss: 2.2952
Epoch [    2/   20] | d_loss: 0.5102 | g_loss: 2.0177
Epoch [    2/   20] | d_loss: 0.2140 | g_loss: 2.4835
Epoch [    2/   20] | d_loss: 0.2881 | g_loss: 3.0662
Epoch [    2/   20] | d_loss: 0.6094 | g_loss: 4.3865
Epoch [    2/   20] | d_loss: 1.0463 | g_loss: 0.5821
Epoch [    3/   20] | d_loss: 0.2581 | g_loss: 3.0789
Epoch [    3/   20] | d_loss: 0.3374 | g_loss: 2.9504
Epoch [    3/   20] | d_loss: 0.4010 | g_loss: 3.7125
Epoch [    3/   20] | d_loss: 0.3573 | g_loss: 3.5461
Epoch [    3/   20] | d_loss: 0.6323 | g_loss: 2.1520
Epoch [    3/   20] | d_loss: 0.2975 | g_loss: 2.7970
Epoch [    3/   20] | d_loss: 0.4401 | g_loss: 2.5194
Epoch [    3/   20] | d_loss: 0.3759 | g_loss: 2.5489
Epoch [    3/   20] | d_loss: 0.1796 | g_loss: 3.3235
Epoch [    3/   20] | d_loss: 0.3249 | g_loss: 3.4348
Epoch [    3/   20] | d_loss: 0.4101 | g_loss: 3.4038
Epoch [    3/   20] | d_loss: 2.2588 | g_loss: 7.1917
Epoch [    3/   20] | d_loss: 0.2429 | g_loss: 2.4250
Epoch [    3/   20] | d_loss: 0.4677 | g_loss: 4.0087
Epoch [    3/   20] | d_loss: 0.4422 | g_loss: 1.9488
Epoch [    3/   20] | d_loss: 1.9693 | g_loss: 1.2937
Epoch [    3/   20] | d_loss: 0.2575 | g_loss: 2.1130
Epoch [    3/   20] | d_loss: 0.3611 | g_loss: 2.7586
Epoch [    4/   20] | d_loss: 0.3684 | g_loss: 3.8329
Epoch [    4/   20] | d_loss: 0.2251 | g_loss: 3.0547
Epoch [    4/   20] | d_loss: 0.3091 | g_loss: 3.9465
Epoch [    4/   20] | d_loss: 0.3589 | g_loss: 2.5990
Epoch [    4/   20] | d_loss: 0.3469 | g_loss: 2.9805
Epoch [    4/   20] | d_loss: 0.3402 | g_loss: 3.6920
Epoch [    4/   20] | d_loss: 0.2985 | g_loss: 3.0195
Epoch [    4/   20] | d_loss: 0.3838 | g_loss: 3.5971
Epoch [    4/   20] | d_loss: 0.2843 | g_loss: 3.7402
Epoch [    4/   20] | d_loss: 0.3281 | g_loss: 2.6685
Epoch [    4/   20] | d_loss: 0.5044 | g_loss: 2.9941
Epoch [    4/   20] | d_loss: 0.2232 | g_loss: 3.1968
Epoch [    4/   20] | d_loss: 0.4342 | g_loss: 2.3485
Epoch [    4/   20] | d_loss: 0.2432 | g_loss: 3.9563
Epoch [    4/   20] | d_loss: 0.2785 | g_loss: 3.1287
Epoch [    4/   20] | d_loss: 0.2519 | g_loss: 2.7374
Epoch [    4/   20] | d_loss: 0.1612 | g_loss: 2.2271
Epoch [    4/   20] | d_loss: 0.2216 | g_loss: 4.2581
Epoch [    5/   20] | d_loss: 0.6022 | g_loss: 5.2644
Epoch [    5/   20] | d_loss: 0.2124 | g_loss: 2.7788
Epoch [    5/   20] | d_loss: 0.3064 | g_loss: 3.9813
Epoch [    5/   20] | d_loss: 0.3976 | g_loss: 3.1309
Epoch [    5/   20] | d_loss: 0.3009 | g_loss: 3.1480
Epoch [    5/   20] | d_loss: 0.3789 | g_loss: 1.2400
Epoch [    5/   20] | d_loss: 0.4666 | g_loss: 1.3366
Epoch [    5/   20] | d_loss: 0.3184 | g_loss: 3.3392
Epoch [    5/   20] | d_loss: 0.1536 | g_loss: 3.5487
Epoch [    5/   20] | d_loss: 2.2436 | g_loss: 0.0102
Epoch [    5/   20] | d_loss: 0.1237 | g_loss: 3.0239
Epoch [    5/   20] | d_loss: 0.2823 | g_loss: 2.0347
Epoch [    5/   20] | d_loss: 0.1712 | g_loss: 3.1507
Epoch [    5/   20] | d_loss: 0.3304 | g_loss: 3.2893
Epoch [    5/   20] | d_loss: 0.3840 | g_loss: 4.3005
Epoch [    5/   20] | d_loss: 0.1458 | g_loss: 4.7049
Epoch [    5/   20] | d_loss: 0.3602 | g_loss: 2.8910
Epoch [    5/   20] | d_loss: 0.1323 | g_loss: 4.2428
Epoch [    6/   20] | d_loss: 0.3126 | g_loss: 2.0454
Epoch [    6/   20] | d_loss: 0.1538 | g_loss: 3.2605
Epoch [    6/   20] | d_loss: 0.6334 | g_loss: 0.8178
Epoch [    6/   20] | d_loss: 0.1981 | g_loss: 3.6217
Epoch [    6/   20] | d_loss: 0.2880 | g_loss: 2.8132
Epoch [    6/   20] | d_loss: 0.3870 | g_loss: 2.4846
Epoch [    6/   20] | d_loss: 0.1616 | g_loss: 3.1717
Epoch [    6/   20] | d_loss: 0.1114 | g_loss: 3.5945
Epoch [    6/   20] | d_loss: 0.2629 | g_loss: 2.4253
Epoch [    6/   20] | d_loss: 0.0867 | g_loss: 3.6061
Epoch [    6/   20] | d_loss: 0.1092 | g_loss: 3.5783
Epoch [    6/   20] | d_loss: 0.1515 | g_loss: 3.1066
Epoch [    6/   20] | d_loss: 0.1975 | g_loss: 3.9695
Epoch [    6/   20] | d_loss: 0.0913 | g_loss: 4.5127
Epoch [    6/   20] | d_loss: 0.5856 | g_loss: 5.3030
Epoch [    6/   20] | d_loss: 0.1791 | g_loss: 3.3090
Epoch [    6/   20] | d_loss: 0.3520 | g_loss: 4.0249
Epoch [    6/   20] | d_loss: 0.0821 | g_loss: 4.6176
Epoch [    7/   20] | d_loss: 1.8079 | g_loss: 6.6244
Epoch [    7/   20] | d_loss: 0.1439 | g_loss: 3.7252
Epoch [    7/   20] | d_loss: 0.1732 | g_loss: 3.0735
Epoch [    7/   20] | d_loss: 0.1663 | g_loss: 4.9652
Epoch [    7/   20] | d_loss: 0.4054 | g_loss: 1.5361
Epoch [    7/   20] | d_loss: 0.1228 | g_loss: 4.0489
Epoch [    7/   20] | d_loss: 0.3069 | g_loss: 3.0065
Epoch [    7/   20] | d_loss: 0.1502 | g_loss: 3.9106
Epoch [    7/   20] | d_loss: 1.3635 | g_loss: 0.5368
Epoch [    7/   20] | d_loss: 0.1281 | g_loss: 4.5330
Epoch [    7/   20] | d_loss: 0.1371 | g_loss: 2.8233
Epoch [    7/   20] | d_loss: 0.1398 | g_loss: 3.4816
Epoch [    7/   20] | d_loss: 0.1472 | g_loss: 2.9651
Epoch [    7/   20] | d_loss: 0.1556 | g_loss: 3.3632
Epoch [    7/   20] | d_loss: 0.3075 | g_loss: 2.1192
Epoch [    7/   20] | d_loss: 0.0968 | g_loss: 3.9695
Epoch [    7/   20] | d_loss: 0.0910 | g_loss: 3.8397
Epoch [    7/   20] | d_loss: 0.7179 | g_loss: 5.9803
Epoch [    8/   20] | d_loss: 0.0424 | g_loss: 4.4372
Epoch [    8/   20] | d_loss: 0.0896 | g_loss: 4.1605
Epoch [    8/   20] | d_loss: 0.2174 | g_loss: 5.0499
Epoch [    8/   20] | d_loss: 0.0652 | g_loss: 3.2943
Epoch [    8/   20] | d_loss: 0.1060 | g_loss: 3.8791
Epoch [    8/   20] | d_loss: 0.0780 | g_loss: 4.3335
Epoch [    8/   20] | d_loss: 0.1254 | g_loss: 3.1223
Epoch [    8/   20] | d_loss: 0.1967 | g_loss: 3.3186
Epoch [    8/   20] | d_loss: 0.1301 | g_loss: 4.1151
Epoch [    8/   20] | d_loss: 0.0830 | g_loss: 4.8445
Epoch [    8/   20] | d_loss: 0.2608 | g_loss: 2.8271
Epoch [    8/   20] | d_loss: 0.1119 | g_loss: 4.5893
Epoch [    8/   20] | d_loss: 0.3671 | g_loss: 5.0487
Epoch [    8/   20] | d_loss: 0.1987 | g_loss: 2.8775
Epoch [    8/   20] | d_loss: 0.0975 | g_loss: 3.9331
Epoch [    8/   20] | d_loss: 0.0533 | g_loss: 4.4056
Epoch [    8/   20] | d_loss: 0.1106 | g_loss: 2.7441
Epoch [    8/   20] | d_loss: 0.0483 | g_loss: 4.0444
Epoch [    9/   20] | d_loss: 0.4065 | g_loss: 1.5017
Epoch [    9/   20] | d_loss: 0.2153 | g_loss: 4.0649
Epoch [    9/   20] | d_loss: 0.0862 | g_loss: 3.6374
Epoch [    9/   20] | d_loss: 0.0468 | g_loss: 4.0638
Epoch [    9/   20] | d_loss: 0.0869 | g_loss: 5.3692
Epoch [    9/   20] | d_loss: 0.3394 | g_loss: 5.6924
Epoch [    9/   20] | d_loss: 0.2919 | g_loss: 3.2044
Epoch [    9/   20] | d_loss: 0.1187 | g_loss: 4.0219
Epoch [    9/   20] | d_loss: 0.2954 | g_loss: 4.6793
Epoch [    9/   20] | d_loss: 0.0664 | g_loss: 4.8897
Epoch [    9/   20] | d_loss: 0.0559 | g_loss: 4.6271
Epoch [    9/   20] | d_loss: 0.0560 | g_loss: 4.1476
Epoch [    9/   20] | d_loss: 0.0613 | g_loss: 4.4777
Epoch [    9/   20] | d_loss: 0.3335 | g_loss: 2.1705
Epoch [    9/   20] | d_loss: 0.0584 | g_loss: 3.7872
Epoch [    9/   20] | d_loss: 0.1220 | g_loss: 3.9204
Epoch [    9/   20] | d_loss: 0.0585 | g_loss: 4.6209
Epoch [    9/   20] | d_loss: 0.0643 | g_loss: 4.6359
Epoch [   10/   20] | d_loss: 0.0362 | g_loss: 4.6476
Epoch [   10/   20] | d_loss: 0.0294 | g_loss: 5.2264
Epoch [   10/   20] | d_loss: 0.0348 | g_loss: 6.0749
Epoch [   10/   20] | d_loss: 0.9174 | g_loss: 1.3330
Epoch [   10/   20] | d_loss: 0.1541 | g_loss: 3.9459
Epoch [   10/   20] | d_loss: 0.1454 | g_loss: 2.0214
Epoch [   10/   20] | d_loss: 0.5492 | g_loss: 1.8234
Epoch [   10/   20] | d_loss: 0.1716 | g_loss: 4.6369
Epoch [   10/   20] | d_loss: 0.0729 | g_loss: 4.2369
Epoch [   10/   20] | d_loss: 0.0632 | g_loss: 4.0842
Epoch [   10/   20] | d_loss: 0.1012 | g_loss: 5.1988
Epoch [   10/   20] | d_loss: 10.6362 | g_loss: 7.9380
Epoch [   10/   20] | d_loss: 1.3067 | g_loss: 0.9736
Epoch [   10/   20] | d_loss: 0.3074 | g_loss: 2.9926
Epoch [   10/   20] | d_loss: 0.2130 | g_loss: 3.3079
Epoch [   10/   20] | d_loss: 0.1538 | g_loss: 2.9729
Epoch [   10/   20] | d_loss: 0.0346 | g_loss: 4.9207
Epoch [   10/   20] | d_loss: 0.0657 | g_loss: 4.7001
Epoch [   11/   20] | d_loss: 0.1366 | g_loss: 4.8172
Epoch [   11/   20] | d_loss: 0.0702 | g_loss: 4.9747
Epoch [   11/   20] | d_loss: 0.0374 | g_loss: 4.5452
Epoch [   11/   20] | d_loss: 0.0732 | g_loss: 5.5488
Epoch [   11/   20] | d_loss: 0.8281 | g_loss: 0.9949
Epoch [   11/   20] | d_loss: 0.2982 | g_loss: 3.7975
Epoch [   11/   20] | d_loss: 0.0832 | g_loss: 4.2944
Epoch [   11/   20] | d_loss: 0.0535 | g_loss: 4.2690
Epoch [   11/   20] | d_loss: 0.0241 | g_loss: 5.0451
Epoch [   11/   20] | d_loss: 0.0239 | g_loss: 5.4374
Epoch [   11/   20] | d_loss: 1.0668 | g_loss: 1.6507
Epoch [   11/   20] | d_loss: 0.0997 | g_loss: 2.8567
Epoch [   11/   20] | d_loss: 0.0968 | g_loss: 4.5450
Epoch [   11/   20] | d_loss: 1.4332 | g_loss: 7.7305
Epoch [   11/   20] | d_loss: 0.5924 | g_loss: 1.6706
Epoch [   11/   20] | d_loss: 0.4711 | g_loss: 5.9267
Epoch [   11/   20] | d_loss: 0.2058 | g_loss: 5.3196
Epoch [   11/   20] | d_loss: 0.1945 | g_loss: 2.6370
Epoch [   12/   20] | d_loss: 0.0366 | g_loss: 3.7850
Epoch [   12/   20] | d_loss: 0.0715 | g_loss: 4.6303
Epoch [   12/   20] | d_loss: 0.0302 | g_loss: 4.6576
Epoch [   12/   20] | d_loss: 0.0411 | g_loss: 3.8231
Epoch [   12/   20] | d_loss: 11.4215 | g_loss: 6.8903
Epoch [   12/   20] | d_loss: 0.0308 | g_loss: 5.1151
Epoch [   12/   20] | d_loss: 0.0873 | g_loss: 4.5265
Epoch [   12/   20] | d_loss: 0.0442 | g_loss: 4.7141
Epoch [   12/   20] | d_loss: 0.0342 | g_loss: 4.5664
Epoch [   12/   20] | d_loss: 0.0551 | g_loss: 4.2502
Epoch [   12/   20] | d_loss: 0.2290 | g_loss: 4.2205
Epoch [   12/   20] | d_loss: 1.2274 | g_loss: 7.3484
Epoch [   12/   20] | d_loss: 0.0315 | g_loss: 5.0526
Epoch [   12/   20] | d_loss: 0.2019 | g_loss: 6.5246
Epoch [   12/   20] | d_loss: 0.2081 | g_loss: 3.5255
Epoch [   12/   20] | d_loss: 0.0756 | g_loss: 4.0890
Epoch [   12/   20] | d_loss: 0.0235 | g_loss: 4.4871
Epoch [   12/   20] | d_loss: 0.0249 | g_loss: 5.6575
Epoch [   13/   20] | d_loss: 1.0538 | g_loss: 1.2199
Epoch [   13/   20] | d_loss: 0.1138 | g_loss: 3.8249
Epoch [   13/   20] | d_loss: 0.0655 | g_loss: 4.1135
Epoch [   13/   20] | d_loss: 0.2688 | g_loss: 7.1522
Epoch [   13/   20] | d_loss: 0.0758 | g_loss: 4.1408
Epoch [   13/   20] | d_loss: 0.1082 | g_loss: 3.8018
Epoch [   13/   20] | d_loss: 0.0897 | g_loss: 4.8337
Epoch [   13/   20] | d_loss: 0.0306 | g_loss: 4.9063
Epoch [   13/   20] | d_loss: 0.0447 | g_loss: 4.2422
Epoch [   13/   20] | d_loss: 0.1986 | g_loss: 8.4908
Epoch [   13/   20] | d_loss: 0.0701 | g_loss: 4.0485
Epoch [   13/   20] | d_loss: 0.0298 | g_loss: 5.4553
Epoch [   13/   20] | d_loss: 0.0222 | g_loss: 6.7034
Epoch [   13/   20] | d_loss: 0.2750 | g_loss: 6.6985
Epoch [   13/   20] | d_loss: 0.4104 | g_loss: 2.1203
Epoch [   13/   20] | d_loss: 0.3537 | g_loss: 6.9217
Epoch [   13/   20] | d_loss: 1.0369 | g_loss: 2.1642
Epoch [   13/   20] | d_loss: 0.0313 | g_loss: 4.5393
Epoch [   14/   20] | d_loss: 0.0502 | g_loss: 4.3150
Epoch [   14/   20] | d_loss: 0.0447 | g_loss: 5.9747
Epoch [   14/   20] | d_loss: 0.0303 | g_loss: 5.2817
Epoch [   14/   20] | d_loss: 0.0211 | g_loss: 4.6926
Epoch [   14/   20] | d_loss: 0.0626 | g_loss: 4.1319
Epoch [   14/   20] | d_loss: 0.0221 | g_loss: 4.6360
Epoch [   14/   20] | d_loss: 0.2538 | g_loss: 4.3052
Epoch [   14/   20] | d_loss: 0.1284 | g_loss: 3.2317
Epoch [   14/   20] | d_loss: 0.0607 | g_loss: 5.5362
Epoch [   14/   20] | d_loss: 0.0400 | g_loss: 5.1305
Epoch [   14/   20] | d_loss: 0.0461 | g_loss: 5.2605
Epoch [   14/   20] | d_loss: 0.0113 | g_loss: 5.2251
Epoch [   14/   20] | d_loss: 0.1055 | g_loss: 3.5958
Epoch [   14/   20] | d_loss: 0.2846 | g_loss: 3.0527
Epoch [   14/   20] | d_loss: 0.0843 | g_loss: 5.2350
Epoch [   14/   20] | d_loss: 0.2065 | g_loss: 8.0867
Epoch [   14/   20] | d_loss: 0.0134 | g_loss: 4.8309
Epoch [   14/   20] | d_loss: 0.0433 | g_loss: 6.3354
Epoch [   15/   20] | d_loss: 0.0453 | g_loss: 5.4577
Epoch [   15/   20] | d_loss: 0.0718 | g_loss: 5.4131
Epoch [   15/   20] | d_loss: 0.6783 | g_loss: 2.1087
Epoch [   15/   20] | d_loss: 0.8370 | g_loss: 4.1956
Epoch [   15/   20] | d_loss: 0.1880 | g_loss: 2.8250
Epoch [   15/   20] | d_loss: 0.0862 | g_loss: 4.6497
Epoch [   15/   20] | d_loss: 0.0599 | g_loss: 4.7886
Epoch [   15/   20] | d_loss: 0.1148 | g_loss: 3.7601
Epoch [   15/   20] | d_loss: 0.1389 | g_loss: 3.9978
Epoch [   15/   20] | d_loss: 0.0477 | g_loss: 4.5356
Epoch [   15/   20] | d_loss: 0.0564 | g_loss: 5.0146
Epoch [   15/   20] | d_loss: 0.0556 | g_loss: 5.4573
Epoch [   15/   20] | d_loss: 0.0639 | g_loss: 3.5775
Epoch [   15/   20] | d_loss: 0.0409 | g_loss: 6.9502
Epoch [   15/   20] | d_loss: 0.0190 | g_loss: 4.9852
Epoch [   15/   20] | d_loss: 5.2883 | g_loss: 3.9293
Epoch [   15/   20] | d_loss: 0.0499 | g_loss: 4.6398
Epoch [   15/   20] | d_loss: 0.0189 | g_loss: 5.3966
Epoch [   16/   20] | d_loss: 0.0270 | g_loss: 4.7387
Epoch [   16/   20] | d_loss: 0.0335 | g_loss: 6.4377
Epoch [   16/   20] | d_loss: 0.0034 | g_loss: 6.1072
Epoch [   16/   20] | d_loss: 0.0500 | g_loss: 5.8209
Epoch [   16/   20] | d_loss: 0.0098 | g_loss: 6.9677
Epoch [   16/   20] | d_loss: 0.0226 | g_loss: 6.6631
Epoch [   16/   20] | d_loss: 0.0293 | g_loss: 6.2700
Epoch [   16/   20] | d_loss: 0.0341 | g_loss: 5.8424
Epoch [   16/   20] | d_loss: 0.0126 | g_loss: 8.0500
Epoch [   16/   20] | d_loss: 0.0040 | g_loss: 6.0602
Epoch [   16/   20] | d_loss: 0.0612 | g_loss: 3.4105
Epoch [   16/   20] | d_loss: 0.0064 | g_loss: 6.6895
Epoch [   16/   20] | d_loss: 0.0049 | g_loss: 7.2504
Epoch [   16/   20] | d_loss: 1.4064 | g_loss: 0.9781
Epoch [   16/   20] | d_loss: 1.8127 | g_loss: 3.9489
Epoch [   16/   20] | d_loss: 0.3158 | g_loss: 2.3938
Epoch [   16/   20] | d_loss: 0.1377 | g_loss: 3.7239
Epoch [   16/   20] | d_loss: 0.6904 | g_loss: 1.7783
Epoch [   17/   20] | d_loss: 0.3208 | g_loss: 5.9080
Epoch [   17/   20] | d_loss: 0.1807 | g_loss: 4.0007
Epoch [   17/   20] | d_loss: 0.1065 | g_loss: 4.1965
Epoch [   17/   20] | d_loss: 0.0272 | g_loss: 5.5360
Epoch [   17/   20] | d_loss: 0.0388 | g_loss: 4.2519
Epoch [   17/   20] | d_loss: 0.1757 | g_loss: 3.8077
Epoch [   17/   20] | d_loss: 0.1436 | g_loss: 2.9925
Epoch [   17/   20] | d_loss: 0.0443 | g_loss: 5.4792
Epoch [   17/   20] | d_loss: 0.0199 | g_loss: 5.6775
Epoch [   17/   20] | d_loss: 0.0791 | g_loss: 4.0401
Epoch [   17/   20] | d_loss: 0.0789 | g_loss: 4.2173
Epoch [   17/   20] | d_loss: 0.1081 | g_loss: 5.0462
Epoch [   17/   20] | d_loss: 0.0363 | g_loss: 5.5640
Epoch [   17/   20] | d_loss: 0.0355 | g_loss: 4.0448
Epoch [   17/   20] | d_loss: 0.0238 | g_loss: 5.6904
Epoch [   17/   20] | d_loss: 0.0086 | g_loss: 6.3412
Epoch [   17/   20] | d_loss: 0.0207 | g_loss: 5.5518
Epoch [   17/   20] | d_loss: 0.0180 | g_loss: 5.6642
Epoch [   18/   20] | d_loss: 0.0079 | g_loss: 5.7917
Epoch [   18/   20] | d_loss: 0.0072 | g_loss: 6.4715
Epoch [   18/   20] | d_loss: 0.0074 | g_loss: 6.5809
Epoch [   18/   20] | d_loss: 0.0147 | g_loss: 5.0512
Epoch [   18/   20] | d_loss: 0.1225 | g_loss: 3.7173
Epoch [   18/   20] | d_loss: 0.0690 | g_loss: 4.1322
Epoch [   18/   20] | d_loss: 0.0276 | g_loss: 6.7996
Epoch [   18/   20] | d_loss: 0.0219 | g_loss: 5.4990
Epoch [   18/   20] | d_loss: 0.0379 | g_loss: 7.2755
Epoch [   18/   20] | d_loss: 0.0126 | g_loss: 4.9835
Epoch [   18/   20] | d_loss: 0.1483 | g_loss: 3.4140
Epoch [   18/   20] | d_loss: 0.0551 | g_loss: 5.3971
Epoch [   18/   20] | d_loss: 0.0743 | g_loss: 5.1957
Epoch [   18/   20] | d_loss: 0.2311 | g_loss: 7.7352
Epoch [   18/   20] | d_loss: 0.1525 | g_loss: 3.8751
Epoch [   18/   20] | d_loss: 0.0388 | g_loss: 5.1979
Epoch [   18/   20] | d_loss: 0.0121 | g_loss: 6.9861
Epoch [   18/   20] | d_loss: 0.0104 | g_loss: 5.7849
Epoch [   19/   20] | d_loss: 0.0347 | g_loss: 6.2739
Epoch [   19/   20] | d_loss: 0.0403 | g_loss: 5.8503
Epoch [   19/   20] | d_loss: 0.0279 | g_loss: 4.7235
Epoch [   19/   20] | d_loss: 0.0261 | g_loss: 4.6225
Epoch [   19/   20] | d_loss: 0.0213 | g_loss: 5.6841
Epoch [   19/   20] | d_loss: 0.0114 | g_loss: 5.6792
Epoch [   19/   20] | d_loss: 0.0175 | g_loss: 5.2884
Epoch [   19/   20] | d_loss: 0.2107 | g_loss: 2.6110
Epoch [   19/   20] | d_loss: 0.1281 | g_loss: 4.7693
Epoch [   19/   20] | d_loss: 0.0495 | g_loss: 4.8789
Epoch [   19/   20] | d_loss: 0.0675 | g_loss: 4.9285
Epoch [   19/   20] | d_loss: 0.0324 | g_loss: 5.6545
Epoch [   19/   20] | d_loss: 0.0838 | g_loss: 4.9898
Epoch [   19/   20] | d_loss: 0.2556 | g_loss: 3.4811
Epoch [   19/   20] | d_loss: 0.0675 | g_loss: 5.3851
Epoch [   19/   20] | d_loss: 0.1427 | g_loss: 4.5205
Epoch [   19/   20] | d_loss: 0.1101 | g_loss: 3.7478
Epoch [   19/   20] | d_loss: 0.4008 | g_loss: 4.7356
Epoch [   20/   20] | d_loss: 0.0862 | g_loss: 4.4108
Epoch [   20/   20] | d_loss: 0.0434 | g_loss: 3.7735
Epoch [   20/   20] | d_loss: 0.0403 | g_loss: 5.4429
Epoch [   20/   20] | d_loss: 0.0145 | g_loss: 4.5721
Epoch [   20/   20] | d_loss: 0.0139 | g_loss: 5.9899
Epoch [   20/   20] | d_loss: 0.0644 | g_loss: 5.3819
Epoch [   20/   20] | d_loss: 0.0678 | g_loss: 6.0349
Epoch [   20/   20] | d_loss: 0.0433 | g_loss: 5.8838
Epoch [   20/   20] | d_loss: 0.4476 | g_loss: 1.9974
Epoch [   20/   20] | d_loss: 0.3603 | g_loss: 2.4428
Epoch [   20/   20] | d_loss: 0.0542 | g_loss: 6.7729
Epoch [   20/   20] | d_loss: 0.1004 | g_loss: 4.6624
Epoch [   20/   20] | d_loss: 0.3363 | g_loss: 2.4089
Epoch [   20/   20] | d_loss: 0.0697 | g_loss: 4.9019
Epoch [   20/   20] | d_loss: 0.1235 | g_loss: 6.1856
Epoch [   20/   20] | d_loss: 0.0701 | g_loss: 4.6239
Epoch [   20/   20] | d_loss: 0.0837 | g_loss: 3.8200
Epoch [   20/   20] | d_loss: 0.0161 | g_loss: 4.6770
Training loss
Plot the training losses for the generator and discriminator, recorded after each epoch.
In [21]:
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
plt.plot(losses.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()
Out[21]:
<matplotlib.legend.Legend at 0x7f36410c5fd0>

Generator samples from training
View samples of images from the generator, and answer a question about the strengths and weaknesses of your trained models.
In [22]:
# helper function for viewing a list of passed in sample images
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = ((img + 1)*255 / (2)).astype(np.uint8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((32,32,3)))
In [23]:
# Load samples from generator, taken while training
with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)
In [24]:
_ = view_samples(-1, samples)

Question: What do you notice about your generated samples and how might you improve this model?
When you answer this question, consider the following factors:
The dataset is biased; it is made of "celebrity" faces that are mostly white
Model size; larger models have the opportunity to learn more features in a data feature space
Optimization strategy; optimizers and number of epochs affect your final result
Answer: As mentioned all the faces in the training set are white. Therefore we could add more diverse faces to balance the dataset.
When I first trained the model there were only 3 layers in the generator and discriminator. The generated faces were not very good. I added more layers and the output has definitley improved. We could add even more layers to learn more complex features and enhance the model. The images in the dataset have a low resolution. Maybe it would be better to have high-resolution images to improve predictions.
I used an Adam optimizer because I often read that it works better than SGD and it's also faster. The model is slow to train so I used 20 epochs. It would probably be better to increase that number.
