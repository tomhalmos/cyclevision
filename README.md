# cyclevision
Computer Vision for Cancer Cell Cycle Research

This Github contains the scripts and algorithms that were developed to implement [Convolutional U-Nets](https://arxiv.org/pdf/1505.04597.pdf) for fully automatic image analysis in cancer cell cycle research. The work was modeled on the double U-Net analysis pipeline used in the [DeLTA](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007673) analysis framework. 


## Train a Network

Training a network requires model initialisation script (U-Net Models), the data.py processing script (Pipeline and Evaluation) and a training script (Training Scripts). To run the three correctly, place them along with the training data in the following folder environment:

> **SegNet**
  > **scripts**
    > data.py
    > model_seg.py
    > train_seg.py
  > **data**
    > img
    > truth

This can be customised by changing the path directories in the DataImport function in data.py. It should be noted that 
