# MyoMapNet Implementation

This folder contains the code to train and test MyoMapNet with several deep learning architectures.

- [main.py](Main.py): This is the main code to train the neural network. It contains the main component such as the different hyperparameters.

- [Architecture.py](Architecture.py): Python code for the first implementation of MyoMapNet with a fully connected neural network

- [Test.py](Test.py): Python code for testing models to generate T1 maps


Architecture

--Contain the implementation of other deep learning architectures: VGG19 - ResNet50, U-Net and ResUnet

Learning curves

--Shows the learning curves obtained during training

## Prerequisite

The MyoMapNet program was implemented using Python `3.6.13` and `Pip 18.1`

All the necessary packages are listed in the requirments.txt file located in the Code directory.

## Create Virtual Enviornment

It is recommended to first create a virtual enviornment prior to running the code.

Create Python venv with Python 3.6:

     python3.6 -m venv myomapnet-venv

Activate venv:

    source myomapnet-venv/bin/activate

Install dependinces:


    pip install -r requirments.txt
    
For more details, please do not hesitate to connet me (Amine Amyar, aamyar@bidmc.harvard.edu)
