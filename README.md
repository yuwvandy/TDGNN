# DPGNN
This repository is an official PyTorch(Geometric) implementation of TDGNN in "Tree Decomposed Graph Neural Network" (CIKM 2021).

For more insights, (empirical and theoretical) analysis, and discussions about Tree Decomposed Graph Neural Networks, please refer to our paper following below.

![](./images/framework.png)

## Requirements
* PyTorch 1.8.1+cu111
* PyTorch Geometric 1.7.0
* NetworkX 2.5.1
* Other frequently-used ML packages

Note that the version of PyTorch and PyTorch Geometric should be compatible and PyTorch Geometric is related to other packages, which requires to be installed beforehand. It is recommended to follow the [installation instruction](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#).

## Run
* To reproduce our results in the following Table, run
```linux
bash run.sh
```
![](./images/CIKM2021-Table2.png)
![](./images/CIKM2021-Table3.png)
