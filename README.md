# PGCGM:
### Physics Guided Generative Adversarial Networks for Generations of Crystal Materials with High Symmetry Constraints

<img src="mainframe.png" height="300px">

Created by Yong Zhao

## Introduction

This repository contains the implementation of generation code of PGCGM.

### Running environment set up

we recommend that you build a virtural environment running the code. Below are steps that can install the dependent packages.

#### create conda virtual environment and activate it
```
conda create -n blm
conda activate blm
```
If there is pip installed, try run below:
```
conda install pip
```

Install `pytorch` from [pytorch web](https://pytorch.org/get-started/previous-versions/) based on your python & cuda version. For example,
```
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
```


## Prerequisites
python 3.8  
pymatgen==2022.0.6  
torch==1.7.1+cu110  
torchaudio==0.7.2  
torchvision==0.8.2+cu110  
numpy==1.19.5  
scipy==1.6.0  

we recommend that you build a virtural environment running the code. Then you just run `sh/gen.sh` to start to generate virtual materials.
