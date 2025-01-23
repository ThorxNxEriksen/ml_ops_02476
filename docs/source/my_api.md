# QuickDraw Model Documentation

## Model Overview

The QuickDraw image classification model is derived from the PyTorch package TIMM with pretrained models. 
Our chosen model is the 'tf_efficientnet_lite0'-model, which is loaded through the code below. 

::: src.quick_draw.model.QuickDrawModel
    :docstring: 
    :members:
    :heading_level: 3

## Training Process

After loading the model, it is trained on 100 images from each of the 10 selected classes. The code for the training process and the associated hyperparameters can be seen in below. 

### Training Functions

::: src.quick_draw.train_wandb
    :docstring:
    :members:
    :heading_level: 3