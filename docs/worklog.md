
# Worklog for GAN Super Resolution Upscaler project
Eirik Vesterkjær, spring 2019


# Current status

2019-04-02

## Things to read up on

1) Progressive Growing of GAN
   1) Dont think this is necessary RN

## Training plan

SRRes: Generator + VGG-19 as feature-extractor (dual layer output)


Train with:
- Regularization, with AdamW
- Data augmentation
  - Noise
  - Data shuffling  
  - Flipping etc
  - Rotation
  - Check the data type, lower precision might give much faster computations
- Optimizer: Try SGD and Adam and AdamW
- Weight init: Fan-in? Kaiming (what is it?)

So whatever model I end up using, I believe that a good approach would be:

1) train GAN architecture with several modifications.
2) Each GAN architecture should have a set max training time (like 4 hours). When I find a good one, train it for longer.

### Loss function

* L1 / L2 pixel loss  ? If model is pretrained this might not be necessary.  
* Relativistic Adversarial loss  
* L1 / L2 of Difference in feature activation  
* L1 / L2 of the feature activation of the difference of the Gen_HR and GR
  * No, not a good idea since some feature activations may be present also when there is no image.

### Architecture



### Hyperparameters

* Training progression (i.e. start on small images and progress, or start directly on large images?)
* Batch size
* Learning rates
  * And scheduling
* Loss penalties
* Loss types (L1/L2)



## Work plan

1) **DONE**: Get an overview of the FULL architecture of the Generator and the Discriminator.  
2) **DONE**: Get an overview of the individual layers - what they do, specifically.  
3) **DONE**: Get an overview of the loss function, and how the gradient of the last hidden layer is computed.  
4) **DONE**: Read the ESRGAN paper again, but this time ensure that I understand it fully.
5) **Skip for now**: Read up on whatever things I do not understand from ESRGAN.
6) **Skip for now**: Read the SRGAN paper again, but this time ensure that I understand it fully.
7) **Skip for now**: Read up on whatever things I do not understand from ESRGAN.
8) **DONE** Determine possible augmentations to the ESRGAN model.
   1) Different generator model
   2) Different discriminator model
   3) Relativistic GAN
      1) *This is actually already used!*
   4) Things that were implemented in other models (EnhanceNet)
   5) Loss function changes 
      1) Loss func proposed by Håkon (Wasserstein distance I believe?)
      2) Features / More feature space output comparison between HR and G(LR)
      3) Other performance metrics like Inception Score / Freichet Inception distance
         1) Might not be applicable to this exact project though.
   6) Transfer learning from a non-GAN model (PSNR-oriented perhaps)
      1) This might be a great project: Looking at the value of adding GAN as a final training phase.
         1) Phase 1: Pretraining PSNR-oriented SR model, maybe using GOIE?
         2) Phase 2: Training SR model with a discriminator - as a GAN
         3) Phase 3: Testing
   7)  Incremental training
   8)  ...
9)  Determine other things that might be interesting to look at
   9)  Stability of GAN (mode collapse etc.)
   10) Uncertainty of GAN (what happens when we add gaussian noise to the input image?)
   11) What features etc. does the generator encode? What happens if we repeatedly zoom in on an image?
   12) Set up my own GAN from top to bottom
10) **DONE**: Set up my own code to train a project, based on BasicSR
11) Changes that now need to be done:
    1)  **DONE**: Save LR images as well -> need name of HR image...
    2)  Add actual validation epochs
        1)  Metrics must be implemented
        2)  The validation loop etc. must be implemented
        3)  Make a function which saves the current output -> move the current "validation" stuff there.
    3)  Go through training loop and compare to ESRGAN
        1)  LFF (tried with 1 so far)
        2)  Specifically, optimizer and such
        3)  Feature extractor
        4)  Other parameters as well!
        5)  Check if batch normalization etc. is used **(no, it's not)**.
    4)  See if LR images from training are different than when no noise etc. is added.
    5)  Make it possible to pass through the full res images.
    6)  Add tensorboard logger for training progress.
    7)  **DONE**: Make progressbar not be SO FUCKING SHIT wrt. printing. statuses - or drop status printing altogether
12) **BUG**: gradient of discriminator seems to be zero after the first iteration.
    1)  Actually, I have no idea of what the fuck is happening.
    2)  The weights of the Discriminator do change... but the grads are zero every validation epoch, except for the first one... huh.

io
- Progressbar-ting?
- Lagre-ting?
- Logger setup?
- Tensorboard setup
- 

## Interesting resources that I have found form this

* https://gluon.mxnet.io/


# Notes



## Setup, preparation, and datasets

CUDA and NVIDIA graphic card drivers were installed, and work. Let's hope they work for PyTorch as well.  

Since I have limited time, my implementation will likely not be vastly different from previous implementations. For this reason, I started by looking at ESRGAN and EnhanceNet.

Although I believed EnhanceNet had sample code, they actually only had the trained network listed, not any training code. ESRGAN however, linked to BasicSR, a repository where the group behind the ESRGAN network host their training code.

I downloaded several datasets. Although I had planned on using the Google Open Images Extended dataset, I soon found that all pictures with faces had them blurred out. This is not desirable for several reasons, so I will either need to use other datasets, or use the GOIE datasets for pretraining only.

ESRGAN provided some tools for image resizing and cropping. I am using those. Might consider using those from enhancenet, as they have *way* more concise code though...

One thing that might be interesting to do, would be to use the methods Håkon mentioned regarding incremental training. 

I had a lot of issues with getting the BasicSR code to run. After some debugging, some fixing of my own mistakes, and installing the newest version of pytorch (due to a bug in the version I had), the basic SR model trains.

As for the SRGAN model, I haven't tested it yet.

Anyways - it looks like basing my desing off of this framework could be a good idea.


## Relativistic GAN

Another thing that might be interesting, is relativistic GAN training. See [here](https://arxiv.org/abs/1807.00734) for the paper. From what I have read, the paper suggests that not only should the generator training seek to maximise the probability that generated outputs are perceived as real - but also reduce the probability that the real outputs are perceived as fake. Their reasoning for this, is the following arguments:
1) Assume that there are equally many generated and real images in a batch. When the generator is fully trained, we should expect D(x_fake) ~ D(x_real) on average. However, when training most GANs, it is (implicitly) assumed that the discriminator does not know this. Training seeks only to maximize D(x_fake). The knowledge of the amount of fakes and reals, is ignored.  
2) When the Jensen-Shannon-divergence is used, we have
    JSD = (1/2*n) * sum_n( log(4) + log[ D(x_r_n) ] + log[ 1 - D(x_f_n)] )
As we can see, this reaches its global minimum at D(x_r_n) == D(x_f_n) == 0.5, where it is 0.
But: If either are *higher* than 0.5, the loss function will not have reached its minimum. So: training should also seek to lower D(x_f)!
3) In the standard GAN, if D is "optimal" in the sense that it distinguishes perfectly, then the real data is completely ignored in the gradient of the discriminator. Note that this is not the case for the Wasserstein distance GAN, if I'm not mistaken, since it is an IPM based GAN where both real and fake data contribute equally to the gradient of the discriminator.





## Getting an overview of G, D, F architecture


I started a training iteration using the ESRGAN model.

While it is training, I'm planning on getting up to speed on how it's constructed, bottom to top.

G = generator
F = Feature extractor
D = Discriminator

### Generator: RRDBNet (16,697,987 parameters)

* Conv2D
* 22x RRDB which consists of
  * 3x ResidualDenseBlock_5C which each consists of,
    * 4x ( Conv2D + LeakyReLU ) (0.2 neg slope)
    * Conv2D
    * 
    * *All these ResidualDenseBlock_5C seem to be with identical layer sizes.*
* Conv2D
* Upsample x 2, nearest neighbour
* Conv2D
* LeakyRLU
* Upsample x 2, nearest neighbour
* Conv2D
* LeakyRLU
* Conv2D
* LeakyRLU
* Conv2D

### Discriminator: VGG discriminator

* Feature extractor
  * Conv2D
  * 9x of the following, with an increasing number of channels (64 -> 512)
    * Conv2D
    * BatchNorm2D 
    * LeakyRLU
* Classifier
  * "Linear" layer: 8192 -> 100
  * LeakyRLU
  * "Linear" layer: 100 -> 1

### Feature extractor for loss function

* 2x Conv2D + ReLU
* MaxPool2D
* 2x Conv2D + ReLU
* MaxPool2D
* 4x Conv2D + ReLU
* MaxPool2D
* 4x Conv2D + ReLU
* MaxPool2D
* 3x Conv2D + ReLU
* Conv2D

## Getting an overview of the individual layers

### Conv2D

Classical Conv layer.  
In G and F, only 3x3 kernels are used, with stride of 1, padding of 1 => no dimension changes from conv2D layers there.

In D, both 3K1S1P and 4K2S1P are used. 4K2S1P results in output dimensions:

( X_in + 2*P - K ) / S + 1 = X_in/2 + 1 + 1 - 2 = X_in / 2

### BatchNorm2D

Batch normalization:  
* Compute the mean and variance of a mini batch {x1, ..., x_m}  
* Using that mean and var, normalize the batch
  * Actually, each x_i is "normalized" by letting z_i = (x_i - mean_batch) / sqrt(var_batch + epsilon)  
* Output: y_i = gamma * z_i + beta  
* The gamma and the beta are the parameters to be learned for this batch norm.

So if I understand correcly, batch norm is essentially a linear transformation done on data that has first been normalized. So the output is of the same dimension as the input.

A resource on batch norm can be found here: https://gluon.mxnet.io/chapter04_convolutional-neural-networks/cnn-batch-norm-scratch.html

So the reasoning behind the batch norm, is that *as training progresses, the inputs to the later layers might change dramatically*. This essentially results in the later layers having to adapt to the new inputs. The idea behind batch normalization is to normalize the mean and var of each feature across the examples in a mini batch. "Empirically, it appears to stabilize the gradient"


### ReLU

Activation is y = max(0, x)

For leaky ReLU, activation is y = max(alpha*x, x) where 0 < alpha < 1


## Getting an overview of ESRGAN's loss function

Model: `SRRaGAN_model.py`: `class SRRaGANModel(BaseModel)`

First of all, the Discriminator is optimized more often than the Generator: `self.D_update_ratio` times as often.

### Generator loss criterions

* Pixel criterion. Train_2 was training with `l1` pixel loss.  
  * `pix_loss = pix_weight * L1_criterion(Generated_HR, GT_HR)`  
* Feature loss. Train_2 was training with `l1` feature loss.  
  * `feat_loss = feat_weight * L1_criterion(F(Generated_HR), F(GT_HR))`  
* GAN loss. Sigmoid layer + Binary cross entropy loss
  * Let `D` be the probability that `GT_HR` is real, and `G` be the probability that `Generated_HR` is real - the probabilities being the outputs of the discriminator. Then the GAN loss is given by,  
  * `0.5 * weight * ( GAN_cri( D - mean(G) , target=False) + GAN_cri( G - mean(D), target=True) )`  
  * This seems to be a slightly modified version of the relativistic gan loss criterion!

### Discriminator loss criterions

* GAN loss, just with inversed labels compared to the generator's GAN loss criterion.
* Also supports WGAN-GP, but it is not used for the ESRGAN.

### ESRGAN's loss function's gradient

How pytorch does this:  
* The network outputs `Gen_HR =  G(LR)`  
* We find `loss_g = f(Gen_HR, HR, D)`  
* We call `loss_g.backward()` to propagate the gradient backwards.  
* We call `optimizer_G.step()` to adjust weights (using momentum + SGD I assume!)

### Comparing SRGAN, ESRGAN, EnhanceNET's loss functions


#### SRGAN

* G: L2 feature loss *after activation* + GAN loss.
* D: Doesn't seem to be specified, but I assume it's something like the inverse of the GAN loss of G, as in ESRGAN.  


#### ESRGAN:

* G: L1 pixel loss + L1 VGG feature loss *before activation* + Relativistic discriminator GAN loss  
* Inverse of G discriminator loss  
* Also uses network interpolation  

#### EnhanceNET  

* G: L2 pixel loss + L2 VGG-19 feature loss *at second and fifth pooling layers* + Texture matching loss based on style transfer + GAN loss 
* D: inverse discriminator GAN loss  
  


## Determining possible changes 


### Generator model

Use ESRGAN's model, but improve on it as described in SRDGAN:
- Adding noise to the LR images?
- Nearest neighbour downsampling?
- Possibly another GAN for downsampling HR images?


### LFF1 vs LFF3

I trained a model with LFF3, and one with LFF1. There was no immediately observable difference wrt. image quality, but training was much faster with LFF1, as a result of the lower amount of parameters.

### Noisy / smoothed labels
Noisy labels can allegedly improve performance and stability. For this reason, I implemented noisy labels. The std. dev. used is 0.05, and the same random noise is used for every sample in each single batch.

Writing this, I realized I had a bug and the random variable is not updated when there is a new batch. Fixed it now. Thanks, me.


### Tensorboard

Having implemented a "working" model, I found that although it did produce images (which improved over time), I had no idea of what was going on in the model.

Adding tensorboard logging allowed me to visualize the training over time, and compare runs to previous ones. What was especially useful, was visualizing the gradients and weights of the first and last layers of D and G. That showed me that the discriminator was not training at all! I fixed this bug in the code.

The result was that the discriminator trained, but although the loss remained relatively low (for now at least), the weights in the final layer of D then appeared to grow (unbounded). Whether this is infact the case remains to be seen. ESRDGAN_train_6_4x_LFF1.



### Kernel size of 5
... for the two HR Conv layers in G, and for the Conv layers with stride of 1 in D.

This *appeared* to give good results. Compare more in the morning.


### Increased number of features + relativistic gan (not avg)

Trying this now..
128 features instead of 64 for both gen and disc.
This requires me to change the batch size.. let's try 96 features instead

relativistic gan instead of relativisticavg


Still have a lot of artifacts! Unsure if this actually helped.



### New upconv

Should try


### Noisy, flipped labels & printing D output distribution
Now with 64 features instead.

Important observation: D outputs are complete garbage.
D weights must be inited with WAY lower values - now the output starts in the 10^2 range..... not good.

### Fixed bug in D, and better initialization

Alright! Now we're talking. I had a .detach placed wrong, which caused the discriminator gradient to only take into account the HR images. This caused it to classify everything as 1. Since the output of D is taken through a sigmoid layer in BCEWithLogitsLoss, this caused the weights in the final layers of D to grow unbounded. Pretty stoked about fixing this!!

Now, let's see how Kaiming initialization helps..! :)
Also using flipped labels and noisy labels. Relativisticavg discriminator, as in ESRGAN


### Still some stability issues

I think it might be better to train the discriminator to convergence,but with added noise on its inputs. As is, it ends up misclassifying HR images over time (SR are always classified as fake, it seems.)

The way forward depends on the stability of the discriminator. 
Here are some of my current challenges:  
- Lack of clarity in the final image  
- Weird tints  
- Weird artifacts and lines.
- Adversary loss is neglible in the generator due to the low weighing in the config. This is the same as for ESRGAN. Should try with far more adversary loss.  



Here are some possible options on what to try next:
- Penalize large weights/grads - regularization
- use another GAN loss function
- Flip labels
- Add noise to the images fed to the discriminator
- Try to fix the vanishing gradients in the discriminator - there is a difference of several orders of magnitude between the first and last layers.  
- Try to use another dataset. I'm not certain that div2k is the most suited one.
- Add pass through of images (to actually _use_ the generator... )
- ReLU or sigmoid or smth on last layer of the Generator. It may be that some pix values now are supposed to be zero, but overflow.
- Training disc more than gen.
- Scale the image pixel value between -1 and 1. Use tanh as the output layer for the generator
- Check out transpose conv, try pixelshuffle
- Try SGD for the discriminator
- Try to use features again!
- Change the upconv!
- Smaller initialization
- Pretrain?
- Go through and rename some vars to make stuff more clear.