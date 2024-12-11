
<div align="center">  
  
# Thermodynamics-informed graph neural networks for Nodal Imposed Displacements

[![Project page](https://img.shields.io/badge/-Project%20page-green)](https://amb.unizar.es/people/lucas/tesan/)
[![Linkedln](https://img.shields.io/badge/-Linkdln%20page-blue)](https://www.linkedin.com/in/lucas-tesan-ingbiozar/)


</div>

## Introduction
This GitHub repository hosts a integration of biomecanical thermodynamics with graph neural networks (GNNs) to power a cutting-edge hepatic digital twin. Our project represents a fusion of computational biology, thermodynamics principles, and advanced machine learning techniques to simulate and understand the intricate dynamics of an hepatic visco-hyperelastic tissue.

<div align="center">
<img src="/resources/liverModel.png" width="1650">
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
- Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, and Peter W. Battaglia. "[Learning mesh-based simulation
with graph networks, 2021.](https://arxiv.org/abs/2010.03409)."
- Alicia Tierz, Iciar Alfaro, David González, Francisco Chinesta, and Elías Cueto. "[Graph neural networks informed
locally by thermodynamics](https://arxiv.org/abs/2405.13093)." arXiv preprint arXiv:2405.13093, May 2024

## Learning procedure

The master database contains 190 simulations across five geometries, with node densities ranging from 680 to 460 per mesh, fixed on the visceral face. Data normalization is performed by standardizing all state variables, which consist of a 12-element vector: three position dimensions, three velocity dimensions, and six components of the Cauchy tensor.

Of the five geometries, four are used for training, resulting in 760 simulations, each with 20 time steps. Each simulation applies nodal displacements to a selection of 1 to 3 nodes, with traction or compression varying between 0.5 cm and 2.5 cm. For validation, 20\% of the simulations from the remaining geometry are used. The primary testing dataset is based on the 190 total simulations of this geometry, including those also used for validation to avoid biases.

Another argument that complements the high response times of this network is its robustness against untrained geometries and meshes, making it highly suitable for precision medicine contexts. For this task, we have complemented the main test database (referred to as **Extra** for this context, as it is the most restrictive) with a set of 190 train simulations, allowing us to compare their performance in a rollout (denoted as **Train**). Additionally, we included another 33 new and untrained simulations within an already trained geometry, designated as **Test** , which functions as an intermediate point between these two testing sets.

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

### Setting Up `wandb` for Experiment Tracking

[Weights & Biases](https://wandb.ai/) (`wandb`) is a tool for tracking machine learning experiments, visualizing metrics, and organizing projects. This guide will walk you through initializing and registering your project with `wandb`.

### Installation

To get started, ensure `wandb` is installed in your environment. You can install it via pip:

```bash
pip install wandb
```

Before using WandB, authenticate by logging in. This is usually done only once on a machine or when starting a new session.

```bash
wandb.login()
```

The model initialization will connect to a project and assign a name to the execution. What is fully implemented in the solver, taking the arguments passed from the main.

```bash
wandb.init(...)
```

### Test pretrained nets

The results of the proyect can be reproduced with the following scripts, found in the `executables/` folder.

```bash
python main.py --dim_hidden 250 --passes 12
```

The `data/` folder includes the database and the pretrained parameters of the networks. The resulting time evolution of the state variables is plotted and saved in .gif format in a generated `outputs/` folder.

### Train a custom net

You can also run your own experiments for the implemented datasets by setting custom parameters manually. Several training examples can be found in the `executables/` folder. The manually trained parameters and output plots are saved in the `outputs/` folder.

```bash
python main.py --train --dim_hidden 250 --passes 12 --lambda_d 5. --max_epoch 5000  --batch_size 8
```
Or just 

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
## Inference performance
Relative error evaluatioón as boxplot for every pseudo-time during inference on test set:

<div>
<img src="/outputs/test_statistics/Test/Test_Unseen_boxplot.png" width="250">
</div>

## 3D Representation and comparisons
Various 3D renderings showing the model's time evolution, along with additional images to compare the final frame inference:

<div>
- Last frame comparison of the test 86th simulation:
</div>
<img src="/resources/render.png" width="1450">
</div>
<div>
- Render of the same test simulation exposing in the color scale the Von Mises stress evolution:
</div>
<img src="/outputs/renders/vm_render_86.gif" width="950">
</div>

## Multi-Graph framework
The complete representation of the system be defined as the superposition of both graphs, forming the multi-graph $G$. Shown in the following example where each color represent a diferent set of graphs overlaped together in a rollout inference.

</div>
<img src="outputs\test_statistics\Test\Test_Unseen_ConnPlot.gif" width="550">
</div>

## Relative and alsolute error for inference on extra dataset

In this section, we present two relative error metrics: one based on the L2 norm and the other on the infinity norm, applied to the three state vectors. Additionally, the absolute error is represented as the root mean square error (RMSE). In all cases, all snapshots from each test subset are considered together.

| Error Type                        | Variable | Error Value   |
|-----------------------------------|----------|---------------|
| **Mean L2 relative error**        | q        | 0.001267      |
|                                   | v        | 0.202649      |
|                                   | sigma    | 0.366082      |
| **Mean inf relative error**       | q        | 0.001038      |
|                                   | v        | 0.042329      |
|                                   | sigma    | 0.019734      |
| **Root mean square error**        | q(m)     | 0.000305      |
|                                   | v(m/s)   | 0.000636      |
|                                   | sigma(Pa)| 1399.189331   |
