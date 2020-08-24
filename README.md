# cyclevision
Computer Vision for Cancer Cell Cycle Research

This Github contains the scripts developed to implement [Convolutional U-Nets](https://arxiv.org/pdf/1505.04597.pdf) for automatic image analysis in cancer cell cycle research. The work was modeled on the [DeLTA](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007673) double U-Net analysis pipeline.

## Python Packages 
The python packages required to train the U-Nets can be installed from the YAML file in the Anaconda Packages folder.

## Train a Network

To train a network set up the following folder environment with the data processing, model architecture and training scripts in the correct location:

  --> **scripts**
  
    -> data.py
    -> model_seg.py
    -> train_seg.py
  --> **data**
  
    -> img
    -> truth

Data importing functions can currently deal with a mixture of 1024x1024 Barr Lab images as well as 696x520 BBBC database images.  
To train the U-Net run the training script from its location in the scripts folder. 
