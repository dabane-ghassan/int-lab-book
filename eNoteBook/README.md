# My INT lab-book
[![forthebadge](https://forthebadge.com/images/badges/made-with-crayons.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/made-with-markdown.svg)](https://forthebadge.com)

- My personal lab-notebook for my internship at the Institut de Neurosciences de la Timone [(INT, Marseille, France)](http://www.int.univ-amu.fr/).

# Tasks
## **Week 1** - *5th to 12th April, 2021*

- [x] Setting up a notebook repo on github. 
- [x] Knocking down a few pages from Andriy Burkov's Hundred page machine learning book. 
- [x] Installing CUDA toolkit and PyTorch. 

- [x] Reading up on Spatial Transformers:
  - [x] Reading the original 2015 paper<sup>2</sup> 
  - [x] Finish up a [presentation](https://youtu.be/6NOQC_fl1hQ). 
  - [x] Reading The PyTorch tutorial [here](https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html). 

- [x] Meeting with the supervisor, wednesday april 7th, 10 a.m. 
- [x] Checking the source code of the What/Where network [repo](https://github.com/laurentperrinet/WhereIsMyMNIST). 
- [x] Checking some Pytorch tutorials. 
  - [x] Official Tutorial [here](https://pytorch.org/tutorials/beginner/basics/intro.html) 
- [x] Thinking about a way to integrate the STN module into the What/Where model &#8594; Reproducing Figure 4 from the original paper <sup>1</sup>. 
- [ ] Preparing slides for the first oral presentation at friday. (post-poned till the 16th) 🚩

- [x] Feeding the MNIST specially modified dataset in a STN. (28x28 with Shift, mask and MotionClouds):
    - [ ] Success rate can't go above 11% on a noisy 28x28, must rethink strategy.
    - [x] Success rate of 94% on a noisy without shift 28x28 dataset. 

- [ ] Factorizing STN : 🚩
  - [x] Separate it into a .py file.
  - [ ] Documentation 🚩

- [x] Starting the Deep learning with Pytorch book to get a better grip of what's happening.

- [ ] Reading up on CNNs.
    - [ ] Colah's blog article [here](http://colah.github.io/posts/2014-07-Conv-Nets-Modular/)

- [x] Integrated LeNet Network with the Spatial Transformer and Test on a 28x28 Noisy MNIST with eccentricity of 2.
- [x] A second meeting with M. Daucé and M. Perrinet
- [x] First BraiNets team meeting

- [ ] Reproduce the classification accuracy map with the same range as the what network.
- [ ] Modifying the CNN architecture of the Localization network (in the STN) to take into account the noisy 128x128 input.



## **Week 2** - *12th to 19th April, 2021*





# Ideas

- Try to feed the STN the log polar input.
- Semi-supervised approach



