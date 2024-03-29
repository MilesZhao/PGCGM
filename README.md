# PGCGM:
### Physics Guided Generative Adversarial Networks for Generations of Crystal Materials with High Symmetry Constraints

<img src="mainframe.png" height="300px">

Created by Yong Zhao

## Introduction

This repository contains the implementation of generation code of PGCGM. If you want to try the code quickly, you run it on Google `Colab`. The colab link is here [PGCGM Colab](https://colab.research.google.com/drive/1VvVl2IO6opptVJkl_fTjC5s6WVKyLh5C?usp=sharing). After click the link, first download the [datamodel.zip](https://github.com/MilesZhao/PGCGM/blob/main/datamodel.zip) file and upload to your colab drive. And then Runtime/runall menu. 

The following installation guide is for running our model on your local machine.

### Running environment set up

we recommend that you build a virtural environment running the code. Below are steps that can install the dependent packages.
The installation has been tested under Ubuntu Linux with Nvidia 2080Ti and Nvidia 3090 GPUs.

#### Step1: Create conda virtual environment and activate it
```
conda create -n pgcgm python=3.7
conda activate pgcgm
```

use 'which' command to check folder of pip3 is located within the actual env ```pgcgm``` folder. It should be somewhere like if you are using miniconda: 
```
which pip3
~/miniconda3/envs/pgcgm/bin/pip3
```
or
```
$HOME/.conda/envs/pgcgm/bin/pip3
```


#### Step2: Then install following packages using:
```
pip3 install pymatgen==2022.0.6
pip3 install pickle5
```
Install `pytorch` from [pytorch web](https://pytorch.org/get-started/previous-versions/) based on your python & cuda version. For example,
For miniconda
```
pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
If your machine has no GPU, use this:
```
pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```


#### Step3: Once you install all those packages, you just run below commands to generate crystal structures:
```
git clone https://github.com/MilesZhao/PGCGM.git
cd PGCGM
./sh/gen.sh
ls ternary_final_cifs
```

Your generated cif files are located in the ```ternary_final_cifs``` folder.
