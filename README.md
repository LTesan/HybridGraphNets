
<div align="center">  
  
# Thermodynamics-informed graph neural networks for Nodal Imposed Displacements

[![Project page](https://img.shields.io/badge/-Project%20page-green)](https://amb.unizar.es/people/lucas/tesan/)
[![Linkedln](https://img.shields.io/badge/-Linkdln%20page-blue)](https://www.linkedin.com/in/lucas-tesan-ingbiozar/)


</div>

## Introduction
This GitHub repository hosts a integration of biomecanical thermodynamics with graph neural networks (GNNs) to power a cutting-edge hepatic digital twin. Our project represents a fusion of computational biology, thermodynamics principles, and advanced machine learning techniques to simulate and understand the intricate dynamics of hepatic systems.

<div align="center">
<img src="/resources/liverModel.png" width="450">
</div>

We introduce an advanced deep learning approach for forecasting the time-based changes in dissipative dynamic systems. Our method leverages geometric and physic biases to enhance accuracy and adaptability in our predictive model. To incorporate geometric insights, we employ Graph Neural Networks, enabling a non-Euclidean geometrical framework with permutation invariant node and edge updates. Additionally, we enforce a thermodynamic bias by training the model to recognize the GENERIC structure of the problem, extending beyond the traditional Hamiltonian formalism to predict a broader range of non-conservative dynamics.

<div align="center">
<img src="/resources/TIGNNs_model.png" width="450">
</div>

Here, you'll find a comprehensive collection of code, models, and resources that drive our Thermodynamics-informed Graph Neural Networks (TIGNNs) framework.

For more information, please refer to the following:

- Tesán, Lucas, González, David and Cueto, Elías.

Original arquitecture:
- Hernández, Quercus and Badías, Alberto and Chinesta, Francisco and Cueto, Elías. "[Thermodynamics-informed graph neural networks](https://ieeexplore.ieee.org/document/9787069)." IEEE Transactions on Artificial Intelligence (2022).

## Setting it up

First, clone the project.

```bash
# clone project
git clone https://github.com/LucasUnizar/TIGNN_NID.git
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
python main.py --n_hidden 2 --dim_hidden 150 --passes 12 --sim 0 --node_num 30 --inference_info
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
| `--output_dir`            | Output data directory                             | Default: `output`                                     |
| `--plot_sim`              | Plot a test simulation                            | `True`, `False`                                       |
| `--inference_info`        | Plot inference info                               | `True`, `False`                                       |

Training Arguments:

|     Argument              |             Description                           | Options                                               |
|---------------------------| ------------------------------------------------- |------------------------------------------------------ |
| `--n_hidden`              | Number of MLP hidden layers                       | Default: `2`                                          |
| `--dim_hidden`            | Dimension of hidden layers                        | Default: `150`                                        |
| `--passes`                | Number of message passing blocks                  | Default: `12`                                         |
| `--lr`                    | Learning rate                                     | Default: `1e-4`                                       |
| `--noise_cons`            | Variance of the constant training noise           | Default: `0`                                          |
| `--noise_prop`            | Variance of the proportional training noise       | Default: `0`                                          |
| `--batch_size`            | Training batch size                               | Default: `2`                                          |
| `--max_epoch`             | Maximum number of training epochs                 | Default: `1500`                                       |
| `--miles`                 | Learning rate scheduler milestones                | Default: `700 1000`                                   |
| `--gamma`                 | Learning rate scheduler decay                     | Default: `5e-1`                                       |
| `--lambda_d`              | Data loss weight                                  | Default: `20`                                         |
| `--node_num`              | Node selection for plots                          | Default: `0`                                          |
| `--sim`                   | Simulation selection for plots                    | Default: `0`                                          |

# Metrics and evaluation
## Boxplots
Relative error evaluatioón as boxplot for every pseudo-time during inference on test set:

<div>
<img src="/outputs/test_statistics/boxplot.png" width="700">
</div>

## 3D Representation and comparisons
Different three-dimensional representations of the time evolution of the model are also drawn comparing both the inference of the network and its ground truth:
<div>
<img src="/outputs/test_statistics/TestLiver_check.gif" width="650">
</div>
Furthermore, a relative nodal error is calculated in a manner similar to the previous representation.
<div>
<img src="/outputs/test_statistics/Liver_error.gif" width="850">
</div>

## Rollout visualization
Finally, a comparative plot between the ground thruth and the network prediction during the time evolution for the selected node.
<div>
<img src="/outputs/model_check/features_check.png" width="4750">
</div>
