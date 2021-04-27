# INT M1 Internship
[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/open-source.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/for-robots.svg)](https://forthebadge.com)

- Supervised by [Emmanuel Daucé](http://emmanuel.dauce.free.fr/) (MC, Ecole Centrale Marseille).
- Description of my internship project:

> In contrast with computer vision, biological vision is characterized by an anisotropic sensor (**The Retina**) as well as the ability to **move** the eyesight to   different locations in the visual scene through ocular **saccades**. To better understand how the human eye analyzes visual scenes, a bio-inspired artificial  vision  model was recently suggested by ***Daucé et al (2020) <sup>1</sup>***.The goal of this master’s internship would be to compare the results obtained by Daucé et   al with some of the more classical attentional computer vision models like the ***Spatial transformer network <sup>2</sup>*** where the visual input undergoes a foveal   deformation.

# Progress

## The Generic Spatial Transformer Network Vs. The What pathway<sup>1</sup>

### Exploring the 28x28 Noisy MNIST dataset. 

> Taking a look at a few examples from the dataset:

![28x28 noisy no shift](figures/data_28x28_noisy_no_shift_.png)

> After transformation with a STN:

![stn results 28x28 noisy no shift](figures/stn_28x28_noisy_no_shift.png)

### Training
- 28x28 Noisy MNIST dataset, with a digit shift standard deviation within the range [0, 15].
- While training, Each 20 epochs, increment the shift standard deviation by 1 (SGD).

> Training statistics:

![training stn 28x28](figures/loss_acc_training_stn_28x28.png)

### Performance

- **Overall results**: *Central* accuracy of **96%** and *general* accuracy of **51%**, compared to **84%** and **34%** in the generic what pathway, respectively.

> Accuracy map comparaison with the generic what pathway from the paper with the same training parameters:

Spatial Transformer Network             |  Generic What pathway <sup>1</sup>
:-------------------------:|:-------------------------:
![acc map stn](figures/stn_28x28_accuracy_map.png)  |  ![acc map what](figures/what_map.png)


> A test on a noisy dataset with a shift standard deviation = 7

![results](figures/results_trained.png)

## The Spatial Transformer Network Vs. The What/Where pathway<sup>1</sup>

### Exploring the 128x128 Noisy MNIST dataset <sup>1</sup>.

> Taking a look at a few examples:

![128x128 noisy shift dataset](figures/data_128x128_noisy_no_shift_.png)

> Some preliminary results with a new architechture that has been trained for **25** epochs on the dataset **without a shift** (the digit is always centered)

![transformed 128x128](figures/preliminary_128x128.png)

> Central accuracy = **98%**
# Papers

[*[1] Emmanuel Daucé, Pierre Albiges, Laurent U. Perrinet; A dual foveal-peripheral visual processing model implements efficient saccade selection. Journal of Vision 2020;20(8):22.*](https://jov.arvojournals.org/article.aspx?articleid=2770680)

[*[2] Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu; Spatial Transformer Networks. arXiv:1506.02025*](https://arxiv.org/abs/1506.02025)

[*[3] Ethan Harris, Mahesan Niranjan, Jonathon Hare; Foveated Convolutions: Improving Spatial Transformer Networks by Modelling the Retina.	1st Workshop on Shared Visual Representations in Human and Machine Intelligence (SVRHM), NeurIPS 2019.*](https://eprints.soton.ac.uk/441204/1/5_CameraReadySubmission_workshop.pdf)

[*[4] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby; An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.	arXiv:2010.11929*](https://arxiv.org/pdf/2010.11929.pdf)

# Miscellaneous

- DeepMind's NIPS 2015 Video experiment on STNs [here](https://drive.google.com/file/d/0B1nQa_sA3W2iN3RQLXVFRkNXN0k/view)
- What/Where model <sup>1</sup> paper github repo [here](https://github.com/laurentperrinet/WhereIsMyMNIST)
- Foveated convulutions <sup>3</sup> paper github repo [here](https://github.com/ethanwharris/foveated-convolutions)
