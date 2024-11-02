
<div align="center">  
  
# Thermodynamics-informed graph neural networks for Nodal Imposed Displacements

[![Project page](https://img.shields.io/badge/-Project%20page-green)](https://amb.unizar.es/people/lucas/tesan/)
[![Linkedln](https://img.shields.io/badge/-Linkdln%20page-blue)](https://www.linkedin.com/in/lucas-tesan-ingbiozar/)


</div>

## Introduction
This GitHub repository hosts a integration of biomecanical thermodynamics with graph neural networks (GNNs) to power a cutting-edge hepatic digital twin. Our project represents a fusion of computational biology, thermodynamics principles, and advanced machine learning techniques to simulate and understand the intricate dynamics of an hepatic visco-hyperelastic tissue.

<div align="center">
<img src="/resources/liverModel.png" width="450">
</div>

We introduce an advanced deep learning approach for forecasting the time-based changes in a dissipative dynamic system. Our method leverages geometric and physic biases to enhance accuracy and adaptability in our predictive model. To incorporate geometric insights, we employ Graph Neural Networks, enabling a non-Euclidean multi-graph framework with permutation invariant node and edge updates. Additionally, we enforce a thermodynamic bias by training the model to recognize the GENERIC structure of the problem, extending beyond the traditional Hamiltonian formalism to predict a broader range of non-conservative dynamics.

The database consists of 760 simulations across 4 different geometries and meshes, ranging between 450 and 700 nodes. For testing purposes, three additional datasets are defined. The primary one, labeled **extra**, represents 190 simulations in an untrained geometry. The **test** set extends this inference to 33 simulations within one of the previously seen geometries. Lastly, the **train** set is a partition of the training data used to compare performance during the rollout.

<div align="center">
<img src="/resources/Gen.png" width="850">
</div>

Here, you'll find a comprehensive collection of code, models, renders and resources that drive our hybrid GNN framework.

For more information, please refer to the following:

- Tesán, Lucas, González, David and Cueto, Elías.

References:
- Hernández, Quercus and Badías, Alberto and Chinesta, Francisco and Cueto, Elías. "[Thermodynamics-informed graph neural networks](https://ieeexplore.ieee.org/document/9787069)." IEEE Transactions on Artificial Intelligence (2022).
- Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, and Peter W. Battaglia. Learning mesh-based simulation
with graph networks, 2021.
- Alicia Tierz, Iciar Alfaro, David González, Francisco Chinesta, and Elías Cueto. Graph neural networks informed
locally by thermodynamics. arXiv preprint arXiv:2405.13093, May 2024.

## Setting it up

First, clone the project.

```bash
# clone project
git clone https://github.com/LucasUnizar/HybridGraphNets.git
cd MeshGraps_NID
```

Then, install the needed dependencies. The code is implemented in [Pytorch](https://pytorch.org). _Note that this has been tested using Python 3.9_.

```bash
# install dependencies
pip install numpy scipy matplotlib torch torch-geometric torch-scatter
 ```

## How to run the code  

### Test pretrained nets

The results of the proyect can be reproduced with the following scripts, found in the `executables/` folder.

```bash
python main.py --n_hidden 2 --dim_hidden 250 --passes 12
```

The `data/` folder includes the database and the pretrained parameters of the networks. The resulting time evolution of the state variables is plotted and saved in .gif format in a generated `outputs/` folder.

### Train a custom net

You can also run your own experiments for the implemented datasets by setting custom parameters manually. Several training examples can be found in the `executables/` folder. The manually trained parameters and output plots are saved in the `outputs/` folder.

```bash
python main.py --train --n_hidden 2 --dim_hidden 150 --passes 12 --max_epoch 1500 --miles 400 700 900 1200
```

General Arguments:

|     Argument              |             Description                           | Options                                               |
|---------------------------| ------------------------------------------------- |------------------------------------------------------ |
| `--train`                 | Train mode                                        | `True`, `False`                                       |
| `--gpu`                   | Enable GPU acceleration                           | `True`, `False`                                       |
| `--pretrained`            | Takes pretrained weights fron 'best_model' file   | `True`, `False`                                       |

Training Arguments:

|     Argument              |             Description                           | Options                                               |
|---------------------------| ------------------------------------------------- |------------------------------------------------------ |
| `--n_hidden`              | Number of MLP hidden layers                       | Default: `2`                                          |
| `--dim_hidden`            | Dimension of hidden layers                        | Default: `150`                                        |
| `--passes`                | Number of message passing blocks                  | Default: `12`                                         |
| `--lr`                    | Learning rate                                     | Default: `1e-4`                                       |
| `--noise_var`             | Variance of the constant training noise           | Default: `[0, 0]`                                     |
| `--batch_size`            | Training batch size                               | Default: `2`                                          |
| `--max_epoch`             | Maximum number of training epochs                 | Default: `1500`                                       |
| `--lambda_d`              | Physics loss weight                               | Default: `5`                                          |

# Metrics and evaluation
## Boxplots
Relative error evaluatioón as boxplot for every pseudo-time during inference on test set:

<div>
<img src="/outputs/test_statistics/Test/Test_Unseen_boxplot.png" width="250">
</div>

## 3D Representation and comparisons
Various 3D renderings showing the model's time evolution, along with additional images to compare the final frame inference:

<div>
<img src="/outputs/renders/mesh_comparison_sim_86.png" width="650">
</div>
<div>
<img src="/outputs/renders/vm_render_86.gif" width="650">
</div>
